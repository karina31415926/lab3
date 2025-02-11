#pragma once

#include "image/bitmap_image.h"

#include <filesystem>

class ImageParser {
public:
	[[nodiscard]] static BitmapImage read_bitmap(const std::filesystem::path& file_path);

	static void write_bitmap(const std::filesystem::path& file_path, const BitmapImage& bitmap);
};
