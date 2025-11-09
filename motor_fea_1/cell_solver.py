def curl_and_div_from_face_vectors(faceB):
    """
    MAC-like curl and divergence for a single cube of unit size.
    faceB: dict with keys {'xp','xm','yp','ym','zp','zm'}
           Each value is an iterable (Bx, By, Bz) giving the field vector
           at that face center:
             xp/xm -> faces orthogonal to +x/-x
             yp/ym -> faces orthogonal to +y/-y
             zp/zm -> faces orthogonal to +z/-z
    Returns: (curl_x, curl_y, curl_z), div
    """
    # Pull components
    Bxp = faceB['xp']; Bxm = faceB['xm']
    Byp = faceB['yp']; Bym = faceB['ym']
    Bzp = faceB['zp']; Bzm = faceB['zm']

    Bx_xp, By_xp, Bz_xp = Bxp
    Bx_xm, By_xm, Bz_xm = Bxm
    Bx_yp, By_yp, Bz_yp = Byp
    Bx_ym, By_ym, Bz_ym = Bym
    Bx_zp, By_zp, Bz_zp = Bzp
    Bx_zm, By_zm, Bz_zm = Bzm

    # MAC curl at cell center (Δ=1):
    # curl_x = ∂Bz/∂y - ∂By/∂z
    curl_x = (Bz_yp - Bz_ym) - (By_zp - By_zm)
    # curl_y = ∂Bx/∂z - ∂Bz/∂x
    curl_y = (Bx_zp - Bx_zm) - (Bz_xp - Bz_xm)
    # curl_z = ∂By/∂x - ∂Bx/∂y
    curl_z = (By_xp - By_xm) - (Bx_yp - Bx_ym)

    # MAC divergence at cell center (Δ=1):
    div = (Bx_xp - Bx_xm) + (By_yp - By_ym) + (Bz_zp - Bz_zm)

    return (curl_x, curl_y, curl_z), div


# ---- quick sanity checks ----

# 1) Linear field B=(x, y, z) at face centers -> curl=0, div=3
ex1 = {
    'xp': ( +0.5, 0.0 , 0.0 ),  # x=+0.5 face
    'xm': ( -0.5, 0.0 , 0.0 ),
    'yp': ( 0.0 , +0.5, 0.0 ),  # y=+0.5 face
    'ym': ( 0.0 , -0.5, 0.0 ),
    'zp': ( 0.0 , 0.0 , +0.5),  # z=+0.5 face
    'zm': ( 0.0 , 0.0 , -0.5),
}
print(curl_and_div_from_face_vectors(ex1))  # -> ((0.0, 0.0, 0.0), 3.0)

# 2) Solid-body swirl around +z: B=(-y, x, 0) at face centers -> curl=(0,0,2), div=0
ex2 = {
    'xp': ( 0.0 , +0.5, 0.0 ),  # at (x=+0.5,y=0)
    'xm': ( 0.0 , -0.5, 0.0 ),
    'yp': ( -0.5, 0.0 , 0.0 ),  # at (x=0,y=+0.5)
    'ym': ( +0.5, 0.0 , 0.0 ),
    'zp': ( 0.0 , 0.0 , 0.0 ),
    'zm': ( 0.0 , 0.0 , 0.0 ),
}
print(curl_and_div_from_face_vectors(ex2))  # -> ((0.0, 0.0, 2.0), 0.0)

