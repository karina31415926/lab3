#include "test.h"
#include "test.cuh"

#include "authors_lab3.h"

#include <cstdint>
#include <string>
#include <vector>


TEST_F(LabTest, testAuthors) {

    const auto authors_a = std::string(AUTHORS_A);
    const auto authors_b = std::string(AUTHORS_B);
    const auto authors_c = std::string(AUTHORS_C);
    const auto authors_d = std::string(AUTHORS_D);
    const auto authors_e = std::string(AUTHORS_E);
    const auto authors_f = std::string(AUTHORS_F);
    const auto authors_g = std::string(AUTHORS_G);
    const auto authors_h = std::string(AUTHORS_H);
    const auto authors_i = std::string(AUTHORS_I);
    const auto authors_j = std::string(AUTHORS_J);
    const auto authors_k = std::string(AUTHORS_K);
    const auto authors_l = std::string(AUTHORS_L);
    const auto authors_m = std::string(AUTHORS_M);
    const auto authors_n = std::string(AUTHORS_N);



    std::string message = "## Autoren: ##\n";
    message += "Aufgabe \n";
    message += "\ta) " + authors_a + "\n";
    message += "\tb) " + authors_b + "\n";
    message += "\tc) " + authors_c + "\n";
    message += "\td) " + authors_d + "\n";
    message += "\te) " + authors_e + "\n";
    message += "\tf) " + authors_f + "\n";
    message += "\tg) " + authors_g + "\n";
    message += "\th) " + authors_h + "\n";
    message += "\tu) " + authors_i + "\n";
    message += "\tj) " + authors_j + "\n";
    message += "\tk) " + authors_k + "\n";
    message += "\tl) " + authors_l + "\n";
    message += "\tm) " + authors_m + "\n";
    message += "\tn) " + authors_n + "\n";
    message += "\n";

    std::cout << message;


    ASSERT_GT(authors_a.length(), 0);
    ASSERT_GT(authors_b.length(), 0);
    ASSERT_GT(authors_c.length(), 0);
    ASSERT_GT(authors_d.length(), 0);
    ASSERT_GT(authors_e.length(), 0);
    ASSERT_GT(authors_f.length(), 0);
    ASSERT_GT(authors_g.length(), 0);
    ASSERT_GT(authors_h.length(), 0);
    ASSERT_GT(authors_i.length(), 0);
    ASSERT_GT(authors_j.length(), 0);
    ASSERT_GT(authors_k.length(), 0);
    ASSERT_GT(authors_l.length(), 0);
    ASSERT_GT(authors_m.length(), 0);
    ASSERT_GT(authors_n.length(), 0);
}

TEST_F(LabTest, testDevices) {
    std::int32_t device_count = 0;
    get_device_count(&device_count);
    ASSERT_GT(device_count, 0);
}

class DataStructureTest : public LabTest{};


int main(int argc, char** argv) {
   ::testing::InitGoogleTest(&argc, argv);

    const auto tests_return_code = RUN_ALL_TESTS();

    return tests_return_code;
}
