# daily_insolation.m (Huybers & Eisenman) for the present-day orbit (kyear = 0, calendar days),
# as used by simulateGlobal/simulateWatercolumn/parametersChemostat when parday light is off.
# The orbital parameters at kyear = 0 are the first row of the Berger (1978) table in that file
# (a spline evaluated at a knot returns the knot value); omega gets +180 degrees as there.

const ORBIT0 = (ecc=0.017236, omega=(101.37 + 180) * pi / 180, epsilon=23.446 * pi / 180)

"""
    daily_insolation(lat, day) -> W/m^2

Daily mean top-of-atmosphere shortwave at latitude `lat` (degrees) on calendar day `day`
(present-day orbit, solar constant 1365 W/m^2).
"""
function daily_insolation(lat_deg::Real, day::Real; ecc=ORBIT0.ecc, omega=ORBIT0.omega, epsilon=ORBIT0.epsilon)
    lat = lat_deg * pi / 180
    delta_lambda_m = (day - 80) * 2 * pi / 365.2422
    beta = (1 - ecc^2)^(1 / 2)
    lambda_m0 = -2 * ((1 / 2 * ecc + 1 / 8 * ecc^3) * (1 + beta) * sin(-omega) -
                      1 / 4 * ecc^2 * (1 / 2 + beta) * sin(-2 * omega) +
                      1 / 8 * ecc^3 * (1 / 3 + beta) * sin(-3 * omega))
    lambda_m = lambda_m0 + delta_lambda_m
    lambda = lambda_m + (2 * ecc - 1 / 4 * ecc^3) * sin(lambda_m - omega) +
             (5 / 4) * ecc^2 * sin(2 * (lambda_m - omega)) + (13 / 12) * ecc^3 * sin(3 * (lambda_m - omega))
    So = 1365
    delta = asin(sin(epsilon) * sin(lambda))
    if abs(lat) >= pi / 2 - abs(delta)          # polar day or night
        Ho = lat * delta > 0 ? Float64(pi) : 0.0
    else
        Ho = acos(-tan(lat) * tan(delta))
    end
    return So / pi * (1 + ecc * cos(lambda - omega))^2 / (1 - ecc^2)^2 *
           (Ho * sin(lat) * sin(delta) + cos(lat) * cos(delta) * sin(Ho))
end
