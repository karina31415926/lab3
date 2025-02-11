#include "test.h"

#include "authors_lab1.h"

#include <cstdint>
#include <string>
#include <vector>


TEST_F(LabTest, testAuthors) {

    const auto authors_two_a = std::string(AUTHORS_TWO_A);
    const auto authors_two_b = std::string(AUTHORS_TWO_B);
    const auto authors_two_c = std::string(AUTHORS_TWO_C);
    const auto authors_two_d = std::string(AUTHORS_TWO_D);
    const auto authors_two_e = std::string(AUTHORS_TWO_E);
    const auto authors_two_f = std::string(AUTHORS_TWO_F);

    const auto authors_three_a = std::string(AUTHORS_THREE_A);
    const auto authors_three_b = std::string(AUTHORS_THREE_B);
    const auto authors_three_c = std::string(AUTHORS_THREE_C);

    const auto authors_four_a = std::string(AUTHORS_FOUR_A);
    const auto authors_four_b = std::string(AUTHORS_FOUR_B);

    const auto authors_five_a = std::string(AUTHORS_FIVE_A);
    const auto authors_five_b = std::string(AUTHORS_FIVE_B);
    const auto authors_five_c = std::string(AUTHORS_FIVE_C);
    const auto authors_five_d = std::string(AUTHORS_FIVE_D);
    const auto authors_five_e = std::string(AUTHORS_FIVE_E);

    std::string message = "## Autoren: ##\n";
    message += "Aufgabe 2\n";
    message += "\ta) " + authors_two_a + "\n";
    message += "\tb) " + authors_two_b + "\n";
    message += "\tc) " + authors_two_c + "\n";
    message += "\td) " + authors_two_d + "\n";
    message += "\te) " + authors_two_e + "\n";
    message += "\tf) " + authors_two_f + "\n";
    message += "\n";
    
    message += "Aufgabe 3\n";
    message += "\ta) " + authors_three_a + "\n";
    message += "\tb) " + authors_three_b + "\n";
    message += "\tc) " + authors_three_c + "\n";
    message += "\n";
    
    message += "Aufgabe 4\n";
    message += "\ta) " + authors_four_a + "\n";    
    message += "\tb) " + authors_four_b + "\n";
    message += "\n";

    message += "Aufgabe 5\n";
    message += "\ta) " + authors_five_a + "\n";
    message += "\tb) " + authors_five_b + "\n";
    message += "\tc) " + authors_five_c + "\n";
    message += "\td) " + authors_five_d + "\n";
    message += "\te) " + authors_five_e + "\n";
    message += "\n";

    std::cout << message;


    ASSERT_GT(authors_two_a.length(), 0);
    ASSERT_GT(authors_two_b.length(), 0);
    ASSERT_GT(authors_two_c.length(), 0);
    ASSERT_GT(authors_two_d.length(), 0);
    ASSERT_GT(authors_two_e.length(), 0);
    ASSERT_GT(authors_two_f.length(), 0);

    ASSERT_GT(authors_three_a.length(), 0);
    ASSERT_GT(authors_three_b.length(), 0);
    ASSERT_GT(authors_three_c.length(), 0);

    ASSERT_GT(authors_four_a.length(), 0);
    ASSERT_GT(authors_four_b.length(), 0);

    ASSERT_GT(authors_five_a.length(), 0);
    ASSERT_GT(authors_five_b.length(), 0);
    ASSERT_GT(authors_five_c.length(), 0);
    ASSERT_GT(authors_five_d.length(), 0);
    ASSERT_GT(authors_five_e.length(), 0);
}


class DataStructureTest : public LabTest{};


int main(int argc, char** argv) {
   ::testing::InitGoogleTest(&argc, argv);

    const auto tests_return_code = RUN_ALL_TESTS();

    return tests_return_code;
}
