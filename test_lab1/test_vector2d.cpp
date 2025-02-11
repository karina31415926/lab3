#include "test.h"

#include "structures/vector2d.h"

#include <exception>
#include <iostream>

class Vector2dTest : public LabTest {};

TEST_F(Vector2dTest, test_two_b){
    // ==
    auto vector1 = Vector2d<double>();
    auto vector2 = Vector2d<double>();
    auto vector3 = Vector2d<double>();
    vector1.set(123.45, 54.321);
    vector2.set(123.45, 54.321);
    ASSERT_TRUE(vector1 == vector2);
    ASSERT_FALSE(vector1 == vector3);
    vector3.set(42.123, 123);
    ASSERT_FALSE(vector1 == vector3);
    vector3.set(123.45, 54.321);
    ASSERT_TRUE(vector1 == vector3);

    // +
    vector1.set(1.0, 4.0);
    vector2.set(2.5, 612.35);
    ASSERT_DOUBLE_EQ(Vector2d<double>(3.5, 616.35)[0], (vector1 + vector2)[0]);
    ASSERT_DOUBLE_EQ(Vector2d<double>(3.5, 616.35)[1], (vector1 + vector2)[1]);
    // -
    ASSERT_DOUBLE_EQ(Vector2d<double>(-1.5, -608.35)[0], (vector1 - vector2)[0]);
    ASSERT_DOUBLE_EQ(Vector2d<double>(-1.5, -608.35)[1], (vector1 - vector2)[1]);
    // *
    ASSERT_DOUBLE_EQ(Vector2d<double>(10.0, 2449.4)[0], (vector2 * 4.0)[0]);
    ASSERT_DOUBLE_EQ(Vector2d<double>(10.0, 2449.4)[1], (vector2 * 4.0)[1]);
    // /
    ASSERT_DOUBLE_EQ(Vector2d<double>(0.5, 122.47)[0], (vector2 / 5.0)[0]);
    ASSERT_DOUBLE_EQ(Vector2d<double>(0.5, 122.47)[1], (vector2 / 5.0)[1]);
    // const
    ASSERT_DOUBLE_EQ(vector1[0], 1.0);
    ASSERT_DOUBLE_EQ(vector1[1], 4.0);
    ASSERT_DOUBLE_EQ(vector2[0], 2.5);
    ASSERT_DOUBLE_EQ(vector2[1], 612.35);
    
}