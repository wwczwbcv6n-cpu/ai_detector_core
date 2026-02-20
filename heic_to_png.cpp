#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio> // For remove()

#include <libheif/heif.h>
#include <png.h> // For libpng

// Function to write raw pixel data to a PNG file
bool write_png(const std::string& filename, int width, int height, int bit_depth, int color_type, const unsigned char* data, int stride) {
    FILE *fp = nullptr;
    png_structp png_ptr = nullptr;
    png_infop info_ptr = nullptr;

    try {
        fp = fopen(filename.c_str(), "wb");
        if (!fp) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return false;
        }

        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png_ptr) {
            std::cerr << "Error: png_create_write_struct failed." << std::endl;
            return false;
        }

        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
            std::cerr << "Error: png_create_info_struct failed." << std::endl;
            return false;
        }

        if (setjmp(png_jmpbuf(png_ptr))) {
            std::cerr << "Error: Error during init_io." << std::endl;
            return false;
        }

        png_init_io(png_ptr, fp);

        if (setjmp(png_jmpbuf(png_ptr))) {
            std::cerr << "Error: Error during header writing." << std::endl;
            return false;
        }

        png_set_IHDR(png_ptr, info_ptr, width, height,
                     bit_depth, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(png_ptr, info_ptr);

        if (setjmp(png_jmpbuf(png_ptr))) {
            std::cerr << "Error: Error during data writing." << std::endl;
            return false;
        }

        // libheif often gives data with different stride than (width * bytes_per_pixel).
        // Ensure row_pointers are correctly managed.
        // Assuming 8-bit depth, bytes_per_pixel is 3 for RGB.
        // The stride from libheif is the correct row pitch.
        std::vector<png_bytep> row_pointers(height);
        for (int y = 0; y < height; y++) {
            row_pointers[y] = (png_bytep)(data + y * stride);
        }

        png_write_image(png_ptr, row_pointers.data());

        if (setjmp(png_jmpbuf(png_ptr))) {
            std::cerr << "Error: Error during end writing." << std::endl;
            return false;
        }

        png_write_end(png_ptr, NULL);

    } catch (const std::exception& e) {
        std::cerr << "Exception in write_png: " << e.what() << std::endl;
        // Clean up resources if an exception occurs, though png_destroy_write_struct should handle most.
    }

    if (fp) fclose(fp);
    // info_ptr and png_ptr are destroyed by png_destroy_write_struct below.
    // If an error occurs before info_ptr is created, png_destroy_write_struct(&png_ptr, (png_infopp)NULL) handles it.
    if (png_ptr) png_destroy_write_struct(&png_ptr, &info_ptr); // Correct destruction with info_ptr

    return true;
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_heic_file> <output_png_file>" << std::endl;
        return 1;
    }

    std::string input_heic_path = argv[1];
    std::string output_png_path = argv[2];

    heif_context* ctx = heif_context_alloc();
    if (!ctx) {
        std::cerr << "Error: heif_context_alloc failed." << std::endl;
        return 1;
    }

    heif_error err = heif_context_read_from_file(ctx, input_heic_path.c_str(), nullptr);
    if (err.code != heif_error_Ok) {
        std::cerr << "Error reading HEIC file: " << err.message << " (code " << err.code << ")" << std::endl;
        heif_context_free(ctx);
        return 1;
    }

    heif_image_handle* handle = nullptr;
    err = heif_context_get_primary_image_handle(ctx, &handle);
    if (err.code != heif_error_Ok) {
        std::cerr << "Error getting primary image handle: " << err.message << " (code " << err.code << ")" << std::endl;
        heif_context_free(ctx);
        return 1;
    }

    heif_image* img = nullptr;
    heif_decoding_options* decode_options = heif_decoding_options_alloc();
    decode_options->convert_hdr_to_8bit = true; // Convert HDR to 8-bit for common display
    decode_options->strict_decoding = false; // Be a bit more lenient
    err = heif_decode_image(handle, &img, heif_colorspace_RGB, heif_chroma_interleaved_RGB, decode_options);
    heif_decoding_options_free(decode_options);
    if (err.code != heif_error_Ok) {
        std::cerr << "Error decoding image: " << err.message << " (code " << err.code << ")" << std::endl;
        heif_image_handle_release(handle);
        heif_context_free(ctx);
        return 1;
    }

    int width = heif_image_get_width(img, heif_channel_interleaved);
    int height = heif_image_get_height(img, heif_channel_interleaved);

    int stride;
    const unsigned char* data = heif_image_get_plane_readonly(img, heif_channel_interleaved, &stride);
    if (!data) {
        std::cerr << "Error: Could not get image data." << std::endl;
        heif_image_release(img);
        heif_image_handle_release(handle);
        heif_context_free(ctx);
        return 1;
    }

    // Assuming 8-bit RGB
    if (!write_png(output_png_path, width, height, 8, PNG_COLOR_TYPE_RGB, data, stride)) {
        std::cerr << "Error: Failed to write PNG file." << std::endl;
        heif_image_release(img);
        heif_image_handle_release(handle);
        heif_context_free(ctx);
        return 1;
    }

    std::cout << "Successfully converted " << input_heic_path << " to " << output_png_path << std::endl;

    heif_image_release(img);
    heif_image_handle_release(handle);
    heif_context_free(ctx);

    return 0;
}