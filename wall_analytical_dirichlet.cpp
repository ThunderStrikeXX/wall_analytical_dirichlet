/**
 * Analytical solution of the 1D transient heat conduction equation
 * in a solid wall with uniform internal heat generation.
 *
 * Domain:
 *   x ∈ [0, L]
 *
 * Governing equation:
 *   ρ c_p ∂T/∂t = k ∂²T/∂x² + Q
 *
 * Boundary conditions (Dirichlet–Dirichlet):
 *   - x = 0 : T = T0 = 350 K
 *   - x = L : T = TL = 300 K
 *
 * Initial condition:
 *   - Uniform temperature field
 *             T(x, 0) = Tinit = 300 K
 *
 * Source term:
 *   - Uniform, constant volumetric heat generation
 *             Q = const [W/m³]
 *
 * Method:
 *   T(x,t) = Ts(x) + θ(x,t)
 *   where Ts solves steady Poisson: k Ts'' + Q = 0 with Ts(0)=T0, Ts(L)=TL
 *   and θ solves diffusion: ρcp θ_t = k θ_xx with θ(0,t)=θ(L,t)=0
 *   and θ(x,0)=Tinit - Ts(x) expanded in sine series.
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
    constexpr double dx = L / (N - 1);
    constexpr double dt = 1;
    constexpr int time_iter = 1000;
    constexpr int harm = 400;
    const double pi = std::acos(-1.0);

    constexpr double k = 20.0;
    constexpr double rho = 7850.0;
    constexpr double cp = 500.0;

    constexpr double T0 = 350.0;            // x=0
    constexpr double TL = 300.0;            // x=L
    constexpr double Tinit = 300.0;         // Initial uniform
    constexpr double Q = 0.0;

    const double alpha = k / (rho * cp);
    const double b = Q / (2.0 * k);         // convenience: Q/(2k)
    const double a = (TL - T0) / L;         // linear slope from BCs

    // Steady solution:
    // Ts(x) = T0 + a x + b (L x - x^2)
    auto Ts = [&](double x) {
        return T0 + a * x + b * (L * x - x * x);
        };

    std::vector<double> T(N, Tinit);
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = i * dx;

    std::ofstream file("wall_analytical_DD.dat");

    // Fourier coefficients B_n for θ(x,0) = Tinit - Ts(x)
    // θ(x,0) = A + B x + C x^2 with:
    //   A = Tinit - T0
    //   B = -(a + b L)
    //   C = b
    // Expand: θ(x,0) = Σ_{n=1..∞} Bn sin(nπx/L)
    // with Bn = (2/L) ∫0^L θ(x,0) sin(nπx/L) dx
    //
    // Closed form (n >= 1, s = (-1)^n):
    //   I0 = ∫ sin(λx) dx       = (1 - s)/λ
    //   I1 = ∫ x sin(λx) dx     = -L s /λ
    //   I2 = ∫ x^2 sin(λx) dx   = -L^2 s /λ + 2(s - 1)/λ^3
    // where λ = nπ/L.
    //
    // Bn = (2/L)[ A I0 + B I1 + C I2 ]

    const double A = Tinit - T0;
    const double B = -(a + b * L);
    const double C = b;

    std::vector<double> lambda(harm);
    std::vector<double> Bn(harm);

    for (int n = 1; n <= harm; ++n) {
        double lam = n * pi / L;
        lambda[n - 1] = lam;

        double s = (n % 2 == 0) ? 1.0 : -1.0;   // (-1)^n
        double I0 = (1.0 - s) / lam;
        double I1 = (-L * s) / lam;
        double I2 = (-L * L * s) / lam + 2.0 * (s - 1.0) / (lam * lam * lam);

        Bn[n - 1] = (2.0 / L) * (A * I0 + B * I1 + C * I2);
    }

    double start = omp_get_wtime();

    for (int it = 0; it < time_iter; ++it) {

        double t = it * dt;

        for (int i = 0; i < N; ++i) {

            double theta = 0.0;
            for (int n = 0; n < harm; ++n) {
                double lam = lambda[n];
                double expo = std::exp(-alpha * lam * lam * t);
                theta += Bn[n] * std::sin(lam * x[i]) * expo;
            }

            T[i] = Ts(x[i]) + theta;
        }

        for (int i = 0; i < N; ++i) file << T[i] << ", ";
        file << "\n";
    }

    double end = omp_get_wtime();
    std::cout << "Execution time: " << end - start << std::endl;

    file.close();
    return 0;
}