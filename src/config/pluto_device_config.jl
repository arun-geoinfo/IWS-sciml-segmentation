# Pluto device config - MINIMAL VERSION
function get_pluto_device(mode::Symbol=:auto)
    # Always return CPU for Pluto safety
    println("ℹ️  Using CPU mode for Pluto compatibility")
    return identity, :cpu
end