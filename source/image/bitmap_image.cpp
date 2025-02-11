#include "image/bitmap_image.h"

#include <exception>

BitmapImage::BitmapImage(const std::uint32_t image_height, const std::uint32_t image_width)
	: height{ image_height }, width{ image_width } {
	if (image_height > 8192) {
		throw std::exception{};
	}

	if (image_height == 0) {
		throw std::exception{};
	}

	if (image_width > 8192) {
		throw std::exception{};
	}

	if (image_width == 0) {
		throw std::exception{};
	}

	pixels.resize(image_height * image_width, BitmapPixel{ 0, 0, 0 });
}

void BitmapImage::set_pixel(const std::uint32_t y_position, const std::uint32_t x_position, const BitmapPixel pixel) {
	if (y_position >= height) {
		throw std::exception{};
	}

	if (x_position >= width) {
		throw std::exception{};
	}

	pixels[y_position * width + x_position] = pixel;
}

BitmapImage::BitmapPixel BitmapImage::get_pixel(const std::uint32_t y_position, const std::uint32_t x_position) const {
	if (y_position >= height) {
		throw std::exception{};
	}

	if (x_position >= width) {
		throw std::exception{};
	}

	return pixels[y_position * width + x_position];
}

std::uint32_t BitmapImage::get_height() const noexcept {
	return height;
}

std::uint32_t BitmapImage::get_width() const noexcept {
	return width;
}

BitmapImage BitmapImage::transpose() const {
	auto transposed_image = BitmapImage(width, height);

	for (auto y = std::uint32_t(0); y < height; y++) {
		for (auto x = std::uint32_t(0); x < width; x++) {
			transposed_image.set_pixel(x, y, get_pixel(y, x));
		}
	}

	return transposed_image;
}
