/**
 * @file test_slcimage.cpp
 * @brief Tests for loading image tiles with memory map
 */

#include "test_util.h"
#include "SlcImage.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

class SlcImageTest : public BackendTest {
protected:
    static constexpr int H = 40, W = 30;
    std::string filename;
    std::vector<image_complex_type> values;

    void SetUp() override
    {
        BackendTest::SetUp();
        if (IsSkipped()) return;
        auto name = std::string("pycuampcor_test_") + ::testing::UnitTest::GetInstance()->current_test_info()->name()
            + "_" + std::to_string(::getpid()) + ".slc";
        filename = (std::filesystem::temp_directory_path() / name).string();
        values.resize(H*W);
        for (int i = 0; i < H*W; i++) { values[i].x = i; values[i].y = -0.5*i; }
        std::ofstream(filename, std::ios::binary).write((const char *)values.data(), H*W*sizeof(image_complex_type));
    }

    void TearDown() override
    {
        if (!filename.empty()) std::filesystem::remove(filename);
        BackendTest::TearDown();
    }
};

TEST_F(SlcImageTest, LoadTiles)
{
    SlcImage image(filename, H, W, sizeof(image_complex_type), 1);
    // tiles in the middle and at the end of the file
    for (auto tile : {std::vector<int>{3, 5, 7, 11}, std::vector<int>{H-4, W-6, 4, 6}}) {
        const int h0 = tile[0], w0 = tile[1], h = tile[2], w = tile[3];
        auto buffer = make<image_complex_type>(h, w);
        image.loadToDevice(buffer->devData, h0, w0, h, w, stream);
        auto r = download(*buffer);
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                EXPECT_EQ(r[i*w+j].x, values[(h0+i)*W + w0+j].x);
                EXPECT_EQ(r[i*w+j].y, values[(h0+i)*W + w0+j].y);
            }
    }
}

TEST_F(SlcImageTest, Errors)
{
    // file smaller than the image size
    EXPECT_THROW(SlcImage(filename, H+1, W, sizeof(image_complex_type), 1), std::runtime_error);
    // missing file
    EXPECT_THROW(SlcImage(filename + ".missing", H, W, sizeof(image_complex_type), 1), std::runtime_error);
    // zero buffer size for mmap
    SlcImage image(filename, H, W, sizeof(image_complex_type), 0);
    auto buffer = make<image_complex_type>(2, 2);
    EXPECT_THROW(image.loadToDevice(buffer->devData, 0, 0, 2, 2, stream), std::runtime_error);
}

} // namespace
