#include<time.h>
#include<cmath>
using namespace std;

class Random {
public:
   //Random(bool pseudo = true);
   Random(int input_seed);
   double random_real();
   int random_integer(int low, int high);
   int poisson(double mean);
   void randomByAvg(double avg,int num);
private:
   int reseed(); //  Re-randomize the seed.
   int seed,multiplier,add_on;
   //  constants for use in arithmetic operations
};

/*
Random::Random(bool pseudo)
{
   if (pseudo) seed = 1;
   else seed = time(NULL) % INT_MAX;
   multiplier = 2743;
   add_on = 5923;
}
*/


Random::Random(int input_seed)
{
   seed = input_seed;
   multiplier = 2743;
   add_on = 5923;
}



int Random::reseed()
{
   seed = seed * multiplier + add_on;
   return seed;
}

double Random::random_real()
{
   double max = INT_MAX + 1.0;  //INT_MAX = (2)31 -1
   double temp = reseed();
   if (temp < 0) temp = temp + max;
   return temp / max;
}

int Random::poisson(double mean)
{
   double limit = exp(-mean);
   double product = random_real();
   int count = 0;
   while (product > limit) {
      count++;
      product *= random_real();
   }
   return count;
}