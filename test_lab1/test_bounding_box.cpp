#include "test.h"

#include "structures/bounding_box.h"

#include <exception>
#include <iostream>

class BoundingBoxTest : public LabTest {};

TEST_F(BoundingBoxTest, test_two_d){
    auto bb = BoundingBox(-15, 15, -30, 30);
    // get_quadrant
    auto qd0 = bb.get_quadrant(0);
    auto qd1 = bb.get_quadrant(1);
    auto qd2 = bb.get_quadrant(2);
    auto qd3 = bb.get_quadrant(3);
    ASSERT_THROW(auto qd0 = bb.get_quadrant(4), std::exception);

    ASSERT_TRUE(qd0.contains(Vector2d<double>(-15, 30)));
    ASSERT_TRUE(qd0.contains(Vector2d<double>(-15, 0)));
    ASSERT_TRUE(qd0.contains(Vector2d<double>(0, 30)));
    ASSERT_TRUE(qd0.contains(Vector2d<double>(0, 0)));
    ASSERT_FALSE(qd0.contains(Vector2d<double>(9, 9)));
    ASSERT_FALSE(qd0.contains(Vector2d<double>(-9, -9)));
    ASSERT_FALSE(qd0.contains(Vector2d<double>(9, -9)));

    ASSERT_TRUE(qd1.contains(Vector2d<double>(0, 30)));
    ASSERT_TRUE(qd1.contains(Vector2d<double>(0, 0)));
    ASSERT_TRUE(qd1.contains(Vector2d<double>(15, 30)));
    ASSERT_TRUE(qd1.contains(Vector2d<double>(15, 0)));
    ASSERT_FALSE(qd1.contains(Vector2d<double>(-9, 9)));
    ASSERT_FALSE(qd1.contains(Vector2d<double>(-9, -9)));
    ASSERT_TRUE(qd2.contains(Vector2d<double>(-15, 0)));
    ASSERT_TRUE(qd2.contains(Vector2d<double>(0, -30)));
    ASSERT_TRUE(qd2.contains(Vector2d<double>(0, 0)));
    ASSERT_FALSE(qd2.contains(Vector2d<double>(-9, 9)));
    ASSERT_FALSE(qd2.contains(Vector2d<double>(9, 9)));
    ASSERT_FALSE(qd2.contains(Vector2d<double>(9, -9)));

    ASSERT_TRUE(qd3.contains(Vector2d<double>(15, -30)));
    ASSERT_TRUE(qd3.contains(Vector2d<double>(5, 0)));
    ASSERT_TRUE(qd3.contains(Vector2d<double>(0, -30)));
    ASSERT_TRUE(qd3.contains(Vector2d<double>(0, 0)));
    ASSERT_FALSE(qd3.contains(Vector2d<double>(-9, 9)));
    ASSERT_FALSE(qd3.contains(Vector2d<double>(9, 9)));
    ASSERT_FALSE(qd3.contains(Vector2d<double>(-9, -9)));
}