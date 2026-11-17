# ============================================================================
# seg_unet_model.jl — U-Net for IW Segmentation (Lux.jl)
#
# Encoder reuses SAR_CNN block pattern (1->32->64->128->128).
# Decoder adds upsampling + skip connections for pixel-level output.
# Returns (mask_pred, state, intermediates) for physics losses.
#
# Spec: specs/segmentation_model.yaml (SEG-001 through SEG-007)
# ============================================================================

module SegUNet

using Lux, LuxCore, NNlib, Random, Statistics

export build_unet, UNetEncoder, UNetDecoder, SegmentationUNet,
       count_parameters, set_unet_training_mode

# ============================================================================
# Encoder block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
# (No pooling inside — pooling is explicit between blocks for skip connections)
# ============================================================================
struct EncoderBlock <: LuxCore.AbstractLuxContainerLayer{(:conv1, :bn1, :conv2, :bn2)}
    conv1::Conv
    bn1::BatchNorm
    conv2::Conv
    bn2::BatchNorm
end

function EncoderBlock(in_ch::Int, out_ch::Int; kernel_size::Int=3)
    pad = kernel_size ÷ 2
    conv1 = Conv((kernel_size, kernel_size), in_ch => out_ch; pad=pad)
    bn1   = BatchNorm(out_ch)
    conv2 = Conv((kernel_size, kernel_size), out_ch => out_ch; pad=pad)
    bn2   = BatchNorm(out_ch)
    return EncoderBlock(conv1, bn1, conv2, bn2)
end

function (eb::EncoderBlock)(x, ps, st)
    y, st_c1 = eb.conv1(x, ps.conv1, st.conv1)
    y, st_b1 = eb.bn1(y, ps.bn1, st.bn1)
    y = relu.(y)
    y, st_c2 = eb.conv2(y, ps.conv2, st.conv2)
    y, st_b2 = eb.bn2(y, ps.bn2, st.bn2)
    y = relu.(y)
    return y, (conv1=st_c1, bn1=st_b1, conv2=st_c2, bn2=st_b2)
end

# ============================================================================
# Decoder block: Upsample -> Concat skip -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU
# ============================================================================
struct DecoderBlock <: LuxCore.AbstractLuxContainerLayer{(:upconv, :conv1, :bn1, :conv2, :bn2)}
    upconv::ConvTranspose
    conv1::Conv
    bn1::BatchNorm
    conv2::Conv
    bn2::BatchNorm
end

function DecoderBlock(in_ch::Int, skip_ch::Int, out_ch::Int)
    # Transposed conv for 2x upsampling
    upconv = ConvTranspose((2, 2), in_ch => out_ch; stride=(2, 2))
    # After concat: out_ch (from upsample) + skip_ch (from encoder)
    conv1 = Conv((3, 3), out_ch + skip_ch => out_ch; pad=1)
    bn1   = BatchNorm(out_ch)
    conv2 = Conv((3, 3), out_ch => out_ch; pad=1)
    bn2   = BatchNorm(out_ch)
    return DecoderBlock(upconv, conv1, bn1, conv2, bn2)
end

function (db::DecoderBlock)(x, skip, ps, st)
    # Upsample
    y, st_up = db.upconv(x, ps.upconv, st.upconv)

    # Crop or pad skip to match spatial dims (handle off-by-one from pooling)
    sy = size(y)
    ss = size(skip)
    if sy[1] != ss[1] || sy[2] != ss[2]
        # Center-crop skip to match upsampled size
        dh = ss[1] - sy[1]
        dw = ss[2] - sy[2]
        h_start = dh ÷ 2 + 1
        w_start = dw ÷ 2 + 1
        skip = skip[h_start:h_start+sy[1]-1, w_start:w_start+sy[2]-1, :, :]
    end

    # Concatenate along channel dimension
    y = cat(y, skip; dims=3)

    # Double conv
    y, st_c1 = db.conv1(y, ps.conv1, st.conv1)
    y, st_b1 = db.bn1(y, ps.bn1, st.bn1)
    y = relu.(y)
    y, st_c2 = db.conv2(y, ps.conv2, st.conv2)
    y, st_b2 = db.bn2(y, ps.bn2, st.bn2)
    y = relu.(y)

    return y, (upconv=st_up, conv1=st_c1, bn1=st_b1, conv2=st_c2, bn2=st_b2)
end

# ============================================================================
# Full U-Net: 4 encoder blocks + bottleneck + 4 decoder blocks + 1x1 output
#
# Encoder (matches SAR_CNN channel pattern):
#   enc1: 1   -> 32   (256x256 -> 256x256, then pool -> 128x128)
#   enc2: 32  -> 64   (128x128 -> 128x128, then pool -> 64x64)
#   enc3: 64  -> 128  (64x64   -> 64x64,   then pool -> 32x32)
#   enc4: 128 -> 128  (32x32   -> 32x32,   then pool -> 16x16)
#
# Bottleneck: 128 -> 256 (16x16)
#
# Decoder:
#   dec4: 256 + 128 -> 128  (16x16  -> 32x32)
#   dec3: 128 + 128 -> 64   (32x32  -> 64x64)
#   dec2: 64  + 64  -> 32   (64x64  -> 128x128)
#   dec1: 32  + 32  -> 32   (128x128 -> 256x256)
#
# Output: 1x1 conv -> sigmoid
# ============================================================================
struct SegmentationUNet <: LuxCore.AbstractLuxContainerLayer{(
    :enc1, :enc2, :enc3, :enc4,
    :bottleneck,
    :dec4, :dec3, :dec2, :dec1,
    :output_conv
)}
    enc1::EncoderBlock
    enc2::EncoderBlock
    enc3::EncoderBlock
    enc4::EncoderBlock
    bottleneck::EncoderBlock
    dec4::DecoderBlock
    dec3::DecoderBlock
    dec2::DecoderBlock
    dec1::DecoderBlock
    output_conv::Conv
end

function SegmentationUNet(; in_channels::Int=1, out_channels::Int=1)
    # Encoder — matches SAR_CNN: 1->32->64->128->128
    enc1 = EncoderBlock(in_channels, 32; kernel_size=3)
    enc2 = EncoderBlock(32, 64; kernel_size=3)
    enc3 = EncoderBlock(64, 128; kernel_size=3)
    enc4 = EncoderBlock(128, 128; kernel_size=3)

    # Bottleneck
    bottleneck = EncoderBlock(128, 256; kernel_size=3)

    # Decoder
    dec4 = DecoderBlock(256, 128, 128)  # upsample 256->128, concat 128 skip, conv to 128
    dec3 = DecoderBlock(128, 128, 64)   # upsample 128->64, concat 128 skip, conv to 64
    dec2 = DecoderBlock(64, 64, 32)     # upsample 64->32, concat 64 skip, conv to 32
    dec1 = DecoderBlock(32, 32, 32)     # upsample 32->32, concat 32 skip, conv to 32

    # 1x1 output conv (no activation — sigmoid applied separately)
    output_conv = Conv((1, 1), 32 => out_channels)

    return SegmentationUNet(enc1, enc2, enc3, enc4, bottleneck,
                            dec4, dec3, dec2, dec1, output_conv)
end

function (m::SegmentationUNet)(x, ps, st)
    # === Encoder ===
    # enc1: (256,256,1,B) -> (256,256,32,B)
    e1, st_e1 = m.enc1(x, ps.enc1, st.enc1)
    p1 = maxpool(e1, (2, 2); stride=(2, 2))  # -> (128,128,32,B)

    # enc2: -> (128,128,64,B)
    e2, st_e2 = m.enc2(p1, ps.enc2, st.enc2)
    p2 = maxpool(e2, (2, 2); stride=(2, 2))  # -> (64,64,64,B)

    # enc3: -> (64,64,128,B)
    e3, st_e3 = m.enc3(p2, ps.enc3, st.enc3)
    p3 = maxpool(e3, (2, 2); stride=(2, 2))  # -> (32,32,128,B)

    # enc4: -> (32,32,128,B)
    e4, st_e4 = m.enc4(p3, ps.enc4, st.enc4)
    p4 = maxpool(e4, (2, 2); stride=(2, 2))  # -> (16,16,128,B)

    # === Bottleneck ===
    b, st_bn = m.bottleneck(p4, ps.bottleneck, st.bottleneck)  # (16,16,256,B)

    # === Decoder with skip connections ===
    d4, st_d4 = m.dec4(b, e4, ps.dec4, st.dec4)   # (32,32,128,B)
    d3, st_d3 = m.dec3(d4, e3, ps.dec3, st.dec3)   # (64,64,64,B)
    d2, st_d2 = m.dec2(d3, e2, ps.dec2, st.dec2)   # (128,128,32,B)
    d1, st_d1 = m.dec1(d2, e1, ps.dec1, st.dec1)   # (256,256,32,B)

    # === Output ===
    logits, st_out = m.output_conv(d1, ps.output_conv, st.output_conv)
    mask_pred = sigmoid_fast.(logits)  # (256,256,1,B) in [0,1]

    new_st = (
        enc1=st_e1, enc2=st_e2, enc3=st_e3, enc4=st_e4,
        bottleneck=st_bn,
        dec4=st_d4, dec3=st_d3, dec2=st_d2, dec1=st_d1,
        output_conv=st_out
    )

    # SEG-007: Return intermediates for physics losses
    intermediates = (enc1=e1, enc2=e2, enc3=e3, enc4=e4, bottleneck=b)

    return mask_pred, new_st, intermediates
end

# ============================================================================
# Utilities
# ============================================================================

"""Build and initialize a U-Net model. Returns (model, params, state)."""
function build_unet(rng::AbstractRNG=Random.default_rng();
                    in_channels::Int=1, out_channels::Int=1)
    model = SegmentationUNet(; in_channels, out_channels)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

"""Count total trainable parameters."""
function count_parameters(ps)
    return _count_params(ps)
end

function _count_params(ps::NamedTuple)
    s = 0
    for k in keys(ps)
        s += _count_params(ps[k])
    end
    return s
end

_count_params(ps::AbstractArray) = length(ps)
_count_params(::Any) = 0

"""Set BatchNorm training mode for all layers."""
function set_unet_training_mode(st::NamedTuple, training::Bool)
    return _set_bn_training(st, training)
end

function _set_bn_training(st::NamedTuple, training::Bool)
    result = Dict{Symbol,Any}()
    for k in keys(st)
        v = st[k]
        if v isa NamedTuple
            if haskey(v, :running_mean)  # This is a BatchNorm state
                result[k] = merge(v, (training=Val(training),))
            else
                result[k] = _set_bn_training(v, training)
            end
        else
            result[k] = v
        end
    end
    return NamedTuple{keys(st)}(Tuple(result[k] for k in keys(st)))
end

end # module SegUNet
