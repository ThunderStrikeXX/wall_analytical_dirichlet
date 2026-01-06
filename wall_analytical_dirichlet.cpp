/**
 * Analytical solution of the 1D transient heat conduction equation
 * in a solid wall with internal heat generation.
 *
 * Domain:
 *   z ∈ [0, L]
 *
 * Governing equation:
 *   ρ c_p ∂T/∂t = k ∂²T/∂z² + Q
 *
 * Boundary conditions (Dirichlet–Dirichlet):
 *   - z = 0 : T = 350 K
 *   - z = L : T = 300 K
 *
 * Initial condition:
 *   - Uniform temperature field
 *             T(z, 0) = 300 K
 *
 * Source term:
 *   - Uniform, constant volumetric heat generation
 *             Q = const [W/m³]
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>

int main() {

    // Physics and domain
    constexpr int N = 100;
    constexpr double L = 1.0;
    constexpr double dz = L / (N - 1);
    constexpr double dt = 1e-3;
    constexpr int time_iter = 1000;
    constexpr int harm = 400;
    const double pi = acos(-1.0);

    constexpr double k = 20.0;
    constexpr double rho = 7850.0;
    constexpr double cp = 500.0;

    constexpr double T0 = 350.0;   // Left boundary
    constexpr double TL = 300.0;   // Right boundary
    constexpr double Q = 1e8;

    std::vector<double> T(N, 300.0);
    std::vector<double> x(N);

    std::ofstream file("wall_analytical_DD.dat");

    for (int i = 0; i < N; ++i)
        x[i] = i * dz;

    // Thermal diffusivity
    const double alpha = k / (rho * cp);

    // Eigenvalues
    std::vector<double> lambda(harm);
    for (int n = 1; n <= harm; ++n)
        lambda[n - 1] = n * pi / L;

    // Fourier coefficients from IC: T(x,0) - Ts(x)
    auto A_n = [&](int n) {
        int m = n + 1;
        double term1 = (2.0 / L) *
            ((300.0 - T0) * (1 - std::cos(m * pi)) / (m * pi));

        double term2 = (2.0 * Q / k) *
            (L * L / std::pow(m * pi, 3)) *
            (1 - std::cos(m * pi));

        return term1 - term2;
        };

    std::vector<double> An(harm);
    for (int n = 0; n < harm; ++n)
        An[n] = A_n(n);

    double start = omp_get_wtime();

    // Time loop
    for (int j = 0; j < time_iter; ++j) {

        double t = j * dt;

        for (int i = 0; i < N; ++i) {

            // Steady-state solution (Dirichlet–Dirichlet + source)
            double Ts =
                T0
                + (TL - T0) * x[i] / L
                + (Q / (2.0 * k)) * (L * x[i] - x[i] * x[i]);

            // Transient correction
            double Tt = 0.0;
            for (int n = 0; n < harm; ++n) {
                double expo = std::exp(-alpha * lambda[n] * lambda[n] * t);
                Tt += An[n] * std::sin(lambda[n] * x[i]) * expo;
            }

            T[i] = Ts + Tt;
        }

        for (int i = 0; i < N; ++i)
            file << T[i] << ", ";
        file << "\n";
    }

    double end = omp_get_wtime();
    std::cout << "Execution time: " << end - start << std::endl;

    file.close();
    return 0;
}
