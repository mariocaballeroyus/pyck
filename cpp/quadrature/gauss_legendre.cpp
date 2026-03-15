#include "gauss_legendre.hpp"

namespace pyck
{

namespace
{

// Precomputed nodes and weights for Gauss-Legendre quadrature (30 digits of precision)

static const StaticVector<double, 1> nodes1 = (StaticVector<double, 1>() << 0.000000000000000000000000000000 ).finished();
static const StaticVector<double, 1> weights1 = (StaticVector<double, 1>() << 2.000000000000000000000000000000 ).finished();

static const StaticVector<double, 2> nodes2 = (StaticVector<double, 2>() << -0.577350269189625764509148780502, 0.577350269189625764509148780502).finished();
static const StaticVector<double, 2> weights2 = (StaticVector<double, 2>() << 1.000000000000000000000000000000, 1.000000000000000000000000000000 ).finished();

static const StaticVector<double, 3> nodes3 = (StaticVector<double, 3>() << -0.774596669241483377035853079956, 0.000000000000000000000000000000, 0.774596669241483377035853079956).finished();
static const StaticVector<double, 3> weights3 = (StaticVector<double, 3>() << 0.555555555555555555555555555556, 0.888888888888888888888888888889, 0.555555555555555555555555555556).finished();

static const StaticVector<double, 4> nodes4 = (StaticVector<double, 4>() << -0.861136311594052575223946488893, -0.339981043584856264802665759103, 0.339981043584856264802665759103, 0.861136311594052575223946488893).finished();
static const StaticVector<double, 4> weights4 = (StaticVector<double, 4>() << 0.347854845137453857373063949222, 0.652145154862546142626936050778, 0.652145154862546142626936050778, 0.347854845137453857373063949222).finished();

static const StaticVector<double, 5> nodes5 = (StaticVector<double, 5>() << -0.906179845938663992797626878299, -0.538469310105683091036314420700, 0.000000000000000000000000000000, 0.538469310105683091036314420700, 0.906179845938663992797626878299).finished();
static const StaticVector<double, 5> weights5 = (StaticVector<double, 5>() << 0.236926885056189087514264040720, 0.478628670499366468041291514836, 0.568888888888888888888888888889, 0.478628670499366468041291514836, 0.236926885056189087514264040720).finished();

static const StaticVector<double, 6> nodes6 = (StaticVector<double, 6>() << -0.932469514203152027812301554494, -0.661209386466264513661399595020, -0.238619186083196908630501721681, 0.238619186083196908630501721681, 0.661209386466264513661399595020, 0.932469514203152027812301554494).finished();
static const StaticVector<double, 6> weights6 = (StaticVector<double, 6>() << 0.171324492379170345040296142173, 0.360761573048138607569833513838, 0.467913934572691047389870343990, 0.467913934572691047389870343990, 0.360761573048138607569833513838, 0.171324492379170345040296142173).finished();

static const StaticVector<double, 7> nodes7 = (StaticVector<double, 7>() << -0.949107912342758524526189684048, -0.741531185599394439863864773281, -0.405845151377397166906606412077, 0.000000000000000000000000000000, 0.405845151377397166906606412077, 0.741531185599394439863864773281, 0.949107912342758524526189684048).finished();
static const StaticVector<double, 7> weights7 = (StaticVector<double, 7>() << 0.129484966168869693270611432679, 0.279705391489276667901467771424, 0.381830050505118944950369775489, 0.417959183673469387755102040816, 0.381830050505118944950369775489, 0.279705391489276667901467771424, 0.129484966168869693270611432679).finished();

static const StaticVector<double, 8> nodes8 = (StaticVector<double, 8>() << -0.960289856497536231683560868569, -0.796666477413626739591553936476, -0.525532409916328985817739049189, -0.183434642495649804939476142360, 0.183434642495649804939476142360, 0.525532409916328985817739049189, 0.796666477413626739591553936476, 0.960289856497536231683560868569).finished();
static const StaticVector<double, 8> weights8 = (StaticVector<double, 8>() << 0.101228536290376259152531354310, 0.222381034453374470544355994426, 0.313706645877887287337962201987, 0.362683783378361982965150449277, 0.362683783378361982965150449277, 0.313706645877887287337962201987, 0.222381034453374470544355994426, 0.101228536290376259152531354310).finished();

} // namespace

template <std::floating_point T, std::size_t d>
GaussLegendre<T, d>::GaussLegendre(std::size_t num_pts)
    : QuadratureRule<T, d>(GaussLegendre<T, 1>(num_pts)) {}

template <std::floating_point T>
GaussLegendre<T, 1>::GaussLegendre(std::size_t num_pts)
{
    if (!this->lookup_reference(num_pts, this->points_, this->weights_))
    {
        this->compute_reference(num_pts, this->points_, this->weights_);
    }
}

template <std::floating_point T>
void GaussLegendre<T, 1>::compute_reference(std::size_t num_pts, 
                                            Vector<T>& nodes, 
                                            Vector<T>& weights) const
{
    if (num_pts == 0)
    {
        nodes.resize(0);
        weights.resize(0);
        return;
    }

    // Resize outputs
    nodes.resize(num_pts, 1);
    weights.resize(num_pts);

    // Build symmetric tridiagonal Jacobi matrix
    Matrix<T> J = Matrix<T>::Zero(num_pts, num_pts);
    for (std::size_t i = 1; i < num_pts; ++i)
    {
        T i_T = static_cast<T>(i);
        T beta = i_T / std::sqrt(static_cast<T>(4.0) * i_T * i_T - static_cast<T>(1.0));
        J(i, i - 1) = beta;
        J(i - 1, i) = beta;
    }

    // Solve Eigenvalue problem
    Eigen::SelfAdjointEigenSolver<Matrix<T>> solver(J);
    nodes = solver.eigenvalues();

    // Extract weights
    Vector<T> v0 = solver.eigenvectors().row(0);
    for (std::size_t i = 0; i < num_pts; ++i)
    {
        weights(i) = static_cast<T>(2.0) * v0(i) * v0(i);
    }
}

template <std::floating_point T>
bool GaussLegendre<T, 1>::lookup_reference(std::size_t num_pts, Vector<T>& nodes, Vector<T>& weights) const
{
    switch (num_pts)
    {
        case 1: nodes = nodes1.template cast<T>(); weights = weights1.template cast<T>(); return true;
        case 2: nodes = nodes2.template cast<T>(); weights = weights2.template cast<T>(); return true;
        case 3: nodes = nodes3.template cast<T>(); weights = weights3.template cast<T>(); return true;
        case 4: nodes = nodes4.template cast<T>(); weights = weights4.template cast<T>(); return true;
        case 5: nodes = nodes5.template cast<T>(); weights = weights5.template cast<T>(); return true;
        case 6: nodes = nodes6.template cast<T>(); weights = weights6.template cast<T>(); return true;
    }

    switch (num_pts)
    {
        case 7: nodes = nodes7.template cast<T>(); weights = weights7.template cast<T>(); return true;
        case 8: nodes = nodes8.template cast<T>(); weights = weights8.template cast<T>(); return true;
        default: return false;
    }
}

// === Template Instantiations ========================================================

template class pyck::GaussLegendre<double, 1>;
template class pyck::GaussLegendre<double, 2>;
template class pyck::GaussLegendre<double, 3>;

#ifdef PYCK_BUILD_SINGLE_PRECISION
template class pyck::GaussLegendre<float, 1>;
template class pyck::GaussLegendre<float, 2>;
template class pyck::GaussLegendre<float, 3>;
#endif

} // namespace pyck
