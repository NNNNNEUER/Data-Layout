#include "Feature.hpp"
#include "Mat.hpp"
#include "Weight.hpp"

void test_blocked_matmul()
{
    int M = 5, K = 5, N = 5;
    auto a = Mat(M, K);
    a.M_rand();
    // a.display();

    auto b = Mat(K, N);
    b.M_rand();

    // b.display();

    auto golden = golden_matmul(a, b);
    std::vector<int> block = {2, 4, 8, 16};
    for (auto M0 : block)
        for (auto K0 : block)
            for (auto N0 : block)
            {
                auto test = matmul(a, b, M0, K0, N0);
                assert(golden == test);
            }

    std::cerr << "Blocked Matmul Passed\n";
}

int main()
{
    // set global parameters
    size_t H = 107, W = 17, Ci = 100;
    size_t H0 = 16, W0 = 8, Ci0 = 16;
    size_t M0 = 8, K0 = 4, N0 = 4;
    size_t Co = 10, Kh = 7, Kw = 7;
    size_t Co0 = 4;
    size_t stride = 1;

    // initialize Feature with random numbers
    Feature f(H, W, Ci);
    srand(time(0));
    for (size_t i = 0; i < H; ++i)
    {
        for (size_t j = 0; j < W; j++)
        {
            for (size_t k = 0; k < Ci; ++k)
            {
                f.at(i, j, k) = i * W + j + rand();
            }
        }
    }

    // make a copy of f and calculate (H1,W1,Ci1) based on (H,W,C) and (H0,W0,Ci0)
    auto original_f = f;
    f = f.pad2Multiple(H0, W0, Ci0);
    auto H1 = f.H() / H0, W1 = f.W() / W0, Ci1 = f.C() / Ci0;

    // initialize Weight with random numbers
    Weight w(Co, Kh, Kw, Ci);
    for (size_t ko = 0; ko < w.Co(); ++ko)
    {
        for (size_t i = 0; i < w.Kh(); ++i)
        {
            for (size_t j = 0; j < w.Kw(); ++j)
            {
                for (size_t ki = 0; ki < w.Ci(); ++ki)
                {
                    w.at(ko, i, j, ki) = ko * w.Ci() * w.Kh() * w.Kw() + ki * w.Kh() * w.Kw() + i * w.Kw() + j + rand();
                }
            }
        }
    }

    // make a copy of w and calculate Co1 based on Co0 and Co
    auto original_w = w;
    w = w.pad2Multiple(Co0, Ci0);
    auto Co1 = w.Co() / Co0;

    // transform Weight to Mat, each col is corresponding to a flatted set of Ci weights, namely, a set for a Co channel
    auto t_w = w.transform();

    // calculate the answer using data_layouted convolution
    Feature ans(f.H(), f.W(), w.Co());
    for (size_t ko = 0; ko < Co1; ++ko)
    {
        for (size_t i = 0; i < H1; ++i)
        {
            for (size_t j = 0; j < W1; ++j)
            {
                for (size_t ki = 0; ki < Ci1; ++ki)
                {

                    // Carry a piece of padded Feature (sub Feature)
                    // std::cerr << ko << " " << i << " " << j << " " << ki << "\n";
                    auto sub = f.subPad(i, j, ki, H0, W0, Ci0, Kh / 2, Kw / 2);
                    // sub.display(0);
                    // sub.display(1);

                    // transform the sub Feature into a matrix (img2col)
                    auto t_sub_f = sub.transform(Kh, Kw, stride);
                    // std::cerr << "Feature Transform\n";
                    //  t_sub_f.display();

                    // Carry a piece of Weight (sub Weight)
                    auto sub_w = w.subWeight(ko, ki, Co0, Ci0);

                    // transform the sub Weight into a matrix (img2col)
                    auto t_sub_w = sub_w.transform();
                    // t_sub_w.display();

                    // img2col matrix multiplication
                    auto res_mat = matmul(t_sub_f, t_sub_w, M0, K0, N0);
                    // std::cerr << "Matmul\n";
                    //  res_mat.display();
                    //  golden_matmul(t_sub_f, t_w).display();

                    // transform the result matrix back into a result feature
                    auto res = invTrans(H0, W0, Co0, res_mat);
                    // std::cerr << "invTransform\n";

                    // accumulate the result feature
                    ans.subAdd(i, j, ko, res);
                }
            }
        }
        std::cerr << "channel Out [" << ko << "] has completed!" << std::endl;
    }

    std::cout << "Checking......" << std::endl;
    ans = ans.range(original_f.H(), original_f.W(), original_w.Co());
    std::cerr << ans.H() << " " << ans.W() << " " << ans.C() << "\n";

    // generate a golden anwser
    auto ref = golden_WconvF(original_f, original_w, stride);
    std::cerr << ref.H() << " " << ref.W() << " " << ref.C() << "\n";

    // check
    assert(ans == ref);
    std::cout << "Congratulations!" << std::endl;
}

// test_blocked_matmul();
/*size_t num = 0;
Mat fea(4, 7);
Mat kel(3, 3);
for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 7; j++)
    {
        fea.at(i, j) = i;
        num += 1;
    }
num = 0;
for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
    {
        kel.at(i, j) = 1;
        // num += 1;
    }

fea.display();
kel.display();

auto res = golden_Matconv(fea, kel, 3);
res.display();*/
/*
    Feature fea(4, 7, 2);
    Weight kel(1, 3, 3, 2);
    Mat mkel(3, 3);
    for (size_t k = 0; k < 2; k++)
    {
        for (size_t i = 0; i < 4; i++)
            for (size_t j = 0; j < 7; j++)
            {
                fea.at(i, j, k) = i + k;
            }
        fea.display(k);
    }
    for (size_t k = 0; k < 2; k++)
    {
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 3; j++)
            {
                kel.at(0, i, j, k) = 1;
                mkel.at(i, j) = kel.at(0, i, j, k);
            }
        mkel.display();
    }

    auto res = golden_WconvF(fea, kel, 2);
    res.display(0);*/
/*
Feature fea(4, 7, 2);
    for (size_t k = 0; k < 2; k++)
    {
        for (size_t i = 0; i < 4; i++)
            for (size_t j = 0; j < 7; j++)
            {
                fea.at(i, j, k) = i + j + 20 * k;
            }
        fea.display(k);
    }
    auto res = fea.transform(3, 3, 2);
    res.display();*/