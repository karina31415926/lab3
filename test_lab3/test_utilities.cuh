#pragma once
bool compare_device_double(void* d_double_ptr1, void* d_double_ptr2);
bool compare_device_doublevector(void* d_ptr1, void* d_ptr2, size_t size);
bool compare_device_double2vector(void* d_ptr1, void* d_ptr2, size_t size);