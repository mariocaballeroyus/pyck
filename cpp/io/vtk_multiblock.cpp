#include "vtk_multiblock.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace pyck::io
{

template <std::floating_point T>
void write_bezier_vtm(const std::string& path,
                      const std::vector<VtkBezierMesh<T>>& blocks,
                      const std::vector<std::string>& block_names,
                      const std::string& title,
                      bool binary)
{
    namespace fs = std::filesystem;

    const fs::path vtm(path);
    const std::string stem = vtm.stem().string();
    const fs::path child_dir = vtm.parent_path() / stem;
    fs::create_directories(child_dir);

    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw std::runtime_error("vtk_export: cannot open '" + path + "' for writing.");

    os << "<?xml version=\"1.0\"?>\n";
    os << "<!-- " << title << " -->\n";
    os << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" "
          "byte_order=\"LittleEndian\">\n";
    os << "  <vtkMultiBlockDataSet>\n";

    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const std::string child = stem + "_" + std::to_string(i) + ".vtu";
        write_bezier_vtu<T>((child_dir / child).string(), {blocks[i]}, title, binary);

        const std::string name = (i < block_names.size() && !block_names[i].empty())
                                ? block_names[i] : ("block" + std::to_string(i));
        // Child path is relative to the .vtm's own directory.
        os << "    <DataSet index=\"" << i << "\" name=\"" << name
           << "\" file=\"" << stem << "/" << child << "\"/>\n";
    }

    os << "  </vtkMultiBlockDataSet>\n";
    os << "</VTKFile>\n";
}

template <std::floating_point T>
void export_bezier_vtm(
    const std::string& path,
    const std::vector<Ptr<Patch<T, 2>>>& patches,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data,
    const std::vector<std::map<std::string, Matrix<T>>>& point_data_extracted,
    const std::vector<std::string>& patch_names,
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

    write_bezier_vtm<T>(path, blocks, patch_names, title, binary);
}

// === Explicit instantiations ========================================================

template void write_bezier_vtm<double>(
    const std::string&, const std::vector<VtkBezierMesh<double>>&,
    const std::vector<std::string>&, const std::string&, bool);
template void export_bezier_vtm<double>(
    const std::string&, const std::vector<Ptr<Patch<double, 2>>>&,
    const std::vector<std::map<std::string, Matrix<double>>>&,
    const std::vector<std::map<std::string, Matrix<double>>>&,
    const std::vector<std::string>&, const std::string&, bool);

} // namespace pyck::io
