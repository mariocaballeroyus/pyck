#include "vtk_export.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "../basis/nurbs.hpp"

namespace pyck::io
{

namespace
{

constexpr std::uint8_t VTK_BEZIER_QUADRILATERAL = 77;

/**
 * Permutation mapping a VTK higher-order-quad ordinal to the lexicographic
 * control index `i + j·(p+1)`: 4 corners, then each edge's interior points in
 * ascending parametric order, then the interior face block (i fastest).
 */
std::vector<Index> bezier_quad_ordering(Index p, Index q)
{
    auto lex = [p](Index i, Index j) { return i + j * (p + 1); };

    std::vector<Index> perm;
    perm.reserve((p + 1) * (q + 1));

    perm.push_back(lex(0, 0));
    perm.push_back(lex(p, 0));
    perm.push_back(lex(p, q));
    perm.push_back(lex(0, q));
    for (Index i = 1; i < p; ++i) perm.push_back(lex(i, 0));  // edge j = 0
    for (Index j = 1; j < q; ++j) perm.push_back(lex(p, j));  // edge i = p
    for (Index i = 1; i < p; ++i) perm.push_back(lex(i, q));  // edge j = q
    for (Index j = 1; j < q; ++j) perm.push_back(lex(0, j));  // edge i = 0
    for (Index j = 1; j < q; ++j)
        for (Index i = 1; i < p; ++i)
            perm.push_back(lex(i, j));                         // interior face

    return perm;
}

/**
 * Apply the per-direction extraction operators to a field stored row-per-CP in
 * u-fastest order (`flat = iu + iv·nu`). `Cu` contracts the u index (fastest),
 * `Cv` the v index. Returns the refined field in the same u-fastest layout.
 */
template <std::floating_point T>
Matrix<T> kron_apply(const Matrix<T>& Cu, const Matrix<T>& Cv,
                     const Matrix<T>& A_old, Index nu_old, Index nv_old)
{
    const Index nu_new = static_cast<Index>(Cu.rows());
    const Index nv_new = static_cast<Index>(Cv.rows());
    const Index k      = static_cast<Index>(A_old.cols());

    // Contract u, one v-line at a time.
    Matrix<T> A_mid(nu_new * nv_old, k);
    for (Index iv = 0; iv < nv_old; ++iv)
        A_mid.middleRows(iv * nu_new, nu_new).noalias() =
            Cu * A_old.middleRows(iv * nu_old, nu_old);

    // Contract v for each fixed (refined) u index.
    Matrix<T> A_new(nu_new * nv_new, k);
    Matrix<T> col(nv_old, k);
    for (Index iu = 0; iu < nu_new; ++iu) {
        for (Index iv = 0; iv < nv_old; ++iv)
            col.row(iv) = A_mid.row(iv * nu_new + iu);
        const Matrix<T> out = Cv * col;  // (nv_new × k)
        for (Index iv = 0; iv < nv_new; ++iv)
            A_new.row(iv * nu_new + iu) = out.row(iv);
    }
    return A_new;
}

/// Extracted per-CP weights: the baked NURBS weights, or all-ones for B-splines.
template <std::floating_point T>
Vector<T> extracted_weights(const Basis<T>& basis)
{
    if (const auto* nb = dynamic_cast<const NURBS<T>*>(&basis))
        return nb->weights();
    return Vector<T>::Ones(basis.num_basis());
}

template <std::floating_point T>
void validate_field(const std::string& name, const Matrix<T>& arr,
                    Index expected_rows, const char* kind)
{
    if (static_cast<Index>(arr.rows()) != expected_rows)
        throw std::runtime_error(
            "vtk_export: " + std::string(kind) + " field '" + name + "' has " +
            std::to_string(arr.rows()) + " rows but " + std::to_string(expected_rows) +
            " were expected.");
}

} // namespace

template <std::floating_point T>
VtkBezierMesh<T> extract_bezier_mesh(
    const Patch<T, 2>& patch,
    const std::map<std::string, Matrix<T>>& point_data,
    const std::map<std::string, Matrix<T>>& point_data_extracted)
{
    const Basis<T>& bu = patch.basis(0);
    const Basis<T>& bv = patch.basis(1);

    auto [bu_ext, Cu] = bu.bezier_extract();
    auto [bv_ext, Cv] = bv.bezier_extract();

    const Index p = bu.degree();
    const Index q = bv.degree();
    const Index nu_old = bu.num_basis(), nv_old = bv.num_basis();
    const Index nu_new = bu_ext->num_basis(), nv_new = bv_ext->num_basis();

    const Index n_eu = (nu_new - 1) / p;
    const Index n_ev = (nv_new - 1) / q;
    if (n_eu * p + 1 != nu_new || n_ev * q + 1 != nv_new)
        throw std::runtime_error(
            "vtk_export: extraction did not yield N_e·p + 1 functions; basis is "
            "not fully C^0-extracted (internal bug).");

    // Geometry and weights pushed onto the extracted basis.
    const Matrix<T> cps_old = patch.control_pts();
    const Matrix<T> cps_new = kron_apply<T>(Cu, Cv, cps_old, nu_old, nv_old);

    const Vector<T> wu = extracted_weights(*bu_ext);
    const Vector<T> wv = extracted_weights(*bv_ext);

    // Extracted fields, in name → (nu_new·nv_new × k) u-fastest layout.
    std::vector<std::string> names;
    std::vector<Matrix<T>>   ext_fields;
    for (const auto& [name, arr] : point_data) {
        validate_field(name, arr, nu_old * nv_old, "point_data");
        names.push_back(name);
        ext_fields.push_back(kron_apply<T>(Cu, Cv, arr, nu_old, nv_old));
    }
    for (const auto& [name, arr] : point_data_extracted) {
        validate_field(name, arr, nu_new * nv_new, "point_data_extracted");
        names.push_back(name);
        ext_fields.push_back(arr);
    }

    const auto perm = bezier_quad_ordering(p, q);
    const Index npc     = (p + 1) * (q + 1);
    const Index n_cells = n_eu * n_ev;
    const Index n_pts   = n_cells * npc;

    VtkBezierMesh<T> mesh;
    mesh.points.resize(n_pts, 3);
    mesh.rational_weights.resize(n_pts);
    mesh.offsets.resize(n_cells);
    mesh.degrees.resize(n_cells, 3);
    mesh.field_names = names;
    mesh.field_values.resize(names.size());
    for (std::size_t f = 0; f < names.size(); ++f)
        mesh.field_values[f].resize(n_pts, ext_fields[f].cols());

    Index cell = 0;
    Index base = 0;
    for (Index ev = 0; ev < n_ev; ++ev) {
        const Index j0 = ev * q;
        for (Index eu = 0; eu < n_eu; ++eu) {
            const Index i0 = eu * p;
            for (Index l = 0; l < npc; ++l) {
                const Index lexn = perm[l];
                const Index li = lexn % (p + 1);
                const Index lj = lexn / (p + 1);
                const Index src = (i0 + li) + (j0 + lj) * nu_new;
                const Index dst = base + l;

                mesh.points.row(dst) = cps_new.row(src);
                mesh.rational_weights(dst) = wu(i0 + li) * wv(j0 + lj);
                for (std::size_t f = 0; f < names.size(); ++f)
                    mesh.field_values[f].row(dst) = ext_fields[f].row(src);
            }
            base += npc;
            mesh.offsets[cell] = base;
            mesh.degrees(cell, 0) = p;
            mesh.degrees(cell, 1) = q;
            mesh.degrees(cell, 2) = 0;
            ++cell;
        }
    }
    return mesh;
}

template <std::floating_point T>
ColMatrix<T, 2> bezier_anchor_params(const Patch<T, 2>& patch)
{
    auto [bu_ext, Cu] = patch.basis(0).bezier_extract();
    auto [bv_ext, Cv] = patch.basis(1).bezier_extract();

    const Vector<T> gu = bu_ext->greville_abscissae();
    const Vector<T> gv = bv_ext->greville_abscissae();
    const Index nu = static_cast<Index>(gu.size());
    const Index nv = static_cast<Index>(gv.size());

    ColMatrix<T, 2> anchors(nu * nv, 2);
    for (Index iv = 0; iv < nv; ++iv)
        for (Index iu = 0; iu < nu; ++iu) {
            const Index r = iu + iv * nu;
            anchors(r, 0) = gu(iu);
            anchors(r, 1) = gv(iv);
        }
    return anchors;
}

// === Serialization ==================================================================

namespace
{

/// One serialized DataArray: its placement, VTK type, and raw little-endian bytes.
struct OutArray
{
    std::string section;  // "PointData" | "CellData" | "Points" | "Cells"
    std::string name;     // empty for the Points coordinate array
    std::string type;     // "Float64" | "Float32" | "Int64" | "Int32" | "UInt8"
    Index       ncomp;
    std::vector<std::byte> bytes;
};

template <typename Scalar, typename Derived>
std::vector<std::byte> bytes_rowmajor(const Eigen::MatrixBase<Derived>& m)
{
    std::vector<Scalar> buf;
    buf.reserve(static_cast<std::size_t>(m.size()));
    for (Index i = 0; i < m.rows(); ++i)
        for (Index j = 0; j < m.cols(); ++j)
            buf.push_back(static_cast<Scalar>(m(i, j)));

    std::vector<std::byte> out(buf.size() * sizeof(Scalar));
    if (!buf.empty())
        std::memcpy(out.data(), buf.data(), out.size());
    return out;
}

template <typename Scalar>
std::vector<std::byte> bytes_from(const std::vector<Scalar>& buf)
{
    std::vector<std::byte> out(buf.size() * sizeof(Scalar));
    if (!buf.empty())
        std::memcpy(out.data(), buf.data(), out.size());
    return out;
}

/// Stream a serialized array's values back out as ASCII (for non-binary output).
void write_ascii_values(std::ostream& os, const OutArray& a)
{
    auto dump = [&](auto sample) {
        using S = decltype(sample);
        const auto* v = reinterpret_cast<const S*>(a.bytes.data());
        const std::size_t n = a.bytes.size() / sizeof(S);
        for (std::size_t i = 0; i < n; ++i) {
            // UInt8 must print as an integer, not a character.
            if constexpr (std::is_same_v<S, std::uint8_t>)
                os << static_cast<unsigned>(v[i]);
            else
                os << v[i];
            os << ((i + 1) % static_cast<std::size_t>(a.ncomp) == 0 ? '\n' : ' ');
        }
    };

    if      (a.type == "Float64") dump(double{});
    else if (a.type == "Float32") dump(float{});
    else if (a.type == "Int64")   dump(std::int64_t{});
    else if (a.type == "Int32")   dump(std::int32_t{});
    else if (a.type == "UInt8")   dump(std::uint8_t{});
}

/// Concatenate per-patch blocks into the flat arrays a single VTU piece needs.
template <std::floating_point T>
std::vector<OutArray> build_arrays(const std::vector<VtkBezierMesh<T>>& blocks)
{
    Index n_pts = 0, n_cells = 0;
    for (const auto& b : blocks) { n_pts += b.num_points(); n_cells += b.num_cells(); }

    ColMatrix<T, 3> points(n_pts, 3);
    Vector<T>       weights(n_pts);
    IndexMatrix     degrees(n_cells, 3);
    std::vector<std::int64_t> connectivity(n_pts);
    std::vector<std::int64_t> offsets(n_cells);
    std::vector<std::uint8_t> types(n_cells, VTK_BEZIER_QUADRILATERAL);

    const std::vector<std::string>& field_names =
        blocks.empty() ? std::vector<std::string>{} : blocks.front().field_names;
    std::vector<Matrix<T>> fields(field_names.size());
    std::vector<Index>     field_cols(field_names.size());
    for (std::size_t f = 0; f < field_names.size(); ++f) {
        field_cols[f] = blocks.front().field_values[f].cols();
        fields[f].resize(n_pts, field_cols[f]);
    }

    Index prow = 0, crow = 0;
    for (const auto& b : blocks) {
        if (b.field_names != field_names)
            throw std::runtime_error(
                "vtk_export: patches disagree on point-data field set.");
        const Index bp = b.num_points();
        const Index bc = b.num_cells();

        points.middleRows(prow, bp)  = b.points;
        weights.segment(prow, bp)    = b.rational_weights;
        degrees.middleRows(crow, bc) = b.degrees;
        for (std::size_t f = 0; f < field_names.size(); ++f) {
            if (b.field_values[f].cols() != field_cols[f])
                throw std::runtime_error(
                    "vtk_export: field '" + field_names[f] +
                    "' has inconsistent component count across patches.");
            fields[f].middleRows(prow, bp) = b.field_values[f];
        }
        for (Index i = 0; i < bp; ++i) connectivity[prow + i] = prow + i;
        for (Index c = 0; c < bc; ++c) offsets[crow + c] = prow + b.offsets[c];

        prow += bp;
        crow += bc;
    }

    const std::string float_type = (sizeof(T) == 4) ? "Float32" : "Float64";

    std::vector<OutArray> arrays;
    for (std::size_t f = 0; f < field_names.size(); ++f)
        arrays.push_back({"PointData", field_names[f], float_type,
                          field_cols[f], bytes_rowmajor<T>(fields[f])});
    arrays.push_back({"PointData", "RationalWeights", float_type, 1,
                      bytes_rowmajor<T>(weights)});
    arrays.push_back({"CellData", "HigherOrderDegrees", "Int32", 3,
                      bytes_rowmajor<std::int32_t>(degrees)});
    arrays.push_back({"Points", "", float_type, 3, bytes_rowmajor<T>(points)});
    arrays.push_back({"Cells", "connectivity", "Int64", 1, bytes_from(connectivity)});
    arrays.push_back({"Cells", "offsets", "Int64", 1, bytes_from(offsets)});
    arrays.push_back({"Cells", "types", "UInt8", 1, bytes_from(types)});
    return arrays;
}

void write_data_array_tag(std::ostream& os, const OutArray& a,
                          bool binary, std::uint64_t offset)
{
    os << "        <DataArray type=\"" << a.type << "\"";
    if (!a.name.empty()) os << " Name=\"" << a.name << "\"";
    if (a.ncomp > 1)     os << " NumberOfComponents=\"" << a.ncomp << "\"";
    if (binary) {
        os << " format=\"appended\" offset=\"" << offset << "\"/>\n";
    } else {
        os << " format=\"ascii\">\n";
        write_ascii_values(os, a);
        os << "        </DataArray>\n";
    }
}

} // namespace

template <std::floating_point T>
void write_bezier_vtu(const std::string& path,
                      const std::vector<VtkBezierMesh<T>>& blocks,
                      const std::string& title,
                      bool binary)
{
    const std::vector<OutArray> arrays = build_arrays(blocks);

    Index n_pts = 0, n_cells = 0;
    for (const auto& b : blocks) { n_pts += b.num_points(); n_cells += b.num_cells(); }

    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw std::runtime_error("vtk_export: cannot open '" + path + "' for writing.");

    // Full round-trip precision for the ASCII path (the default 6 sig. figs is lossy).
    os.precision(sizeof(T) == 4 ? 9 : 17);

    os << "<?xml version=\"1.0\"?>\n";
    os << "<!-- " << title << " -->\n";
    os << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
          "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    os << "  <UnstructuredGrid>\n";
    os << "    <Piece NumberOfPoints=\"" << n_pts
       << "\" NumberOfCells=\"" << n_cells << "\">\n";

    std::uint64_t offset = 0;
    auto emit_section = [&](const char* open, const char* section) {
        os << "      <" << open << ">\n";
        for (const auto& a : arrays) {
            if (a.section != section) continue;
            write_data_array_tag(os, a, binary, offset);
            if (binary) offset += sizeof(std::uint64_t) + a.bytes.size();
        }
        os << "      </" << open << ">\n";
    };

    emit_section("PointData", "PointData");
    emit_section("CellData", "CellData");
    emit_section("Points", "Points");
    emit_section("Cells", "Cells");

    os << "    </Piece>\n";
    os << "  </UnstructuredGrid>\n";

    if (binary) {
        os << "  <AppendedData encoding=\"raw\">\n_";
        for (const auto& a : arrays) {
            const std::uint64_t nbytes = a.bytes.size();
            os.write(reinterpret_cast<const char*>(&nbytes), sizeof(nbytes));
            if (!a.bytes.empty())
                os.write(reinterpret_cast<const char*>(a.bytes.data()),
                         static_cast<std::streamsize>(a.bytes.size()));
        }
        os << "\n  </AppendedData>\n";
    }

    os << "</VTKFile>\n";
}

template <std::floating_point T>
void export_bezier_vtu(
    const std::string& path,
    const std::vector<Ptr<Patch<T, 2>>>& patches,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data_extracted,
    const std::string& title,
    bool binary)
{
    if (point_data.size() != patches.size() ||
        point_data_extracted.size() != patches.size())
        throw std::runtime_error(
            "vtk_export: per-patch field lists must match the patch count.");

    std::vector<VtkBezierMesh<T>> blocks;
    blocks.reserve(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i)
        blocks.push_back(extract_bezier_mesh<T>(
            *patches[i], point_data[i], point_data_extracted[i]));

    write_bezier_vtu<T>(path, blocks, title, binary);
}

// === Explicit instantiations ========================================================

template VtkBezierMesh<double> extract_bezier_mesh<double>(
    const Patch<double, 2>&,
    const std::map<std::string, Matrix<double>>&,
    const std::map<std::string, Matrix<double>>&);
template ColMatrix<double, 2> bezier_anchor_params<double>(const Patch<double, 2>&);
template void write_bezier_vtu<double>(
    const std::string&, const std::vector<VtkBezierMesh<double>>&,
    const std::string&, bool);
template void export_bezier_vtu<double>(
    const std::string&, const std::vector<Ptr<Patch<double, 2>>>&,
    const std::vector<std::map<std::string, Matrix<double>>>&,
    const std::vector<std::map<std::string, Matrix<double>>>&,
    const std::string&, bool);

} // namespace pyck::io
