
/*
 * UNIT TEST
 */

#include <cstdlib>
#include <iostream>
#include <map>
#include <valarray>

#include "stats_utils.hpp"

#include "../unit_test/Catch/catch.hpp"


void StatsOutput() {

  constexpr size_t val_size = 100;
  constexpr long tm_range = 2000;
  constexpr long halflife = 100;
  std::map<long, double> values;

  printf("tm,update\n");
  /*for (size_t ii = 0; ii<val_size; ++ii) {
    long tm = std::rand()%tm_range;
    double val{double(tm%2)};
    values.insert({tm, val});
    printf("%ld,%f\n", tm, val);
  }*/

  double corr = 0.5;
  double last_val = 0.0;
  for (size_t ii = 0; ii<tm_range; ++ii) {
      double renov = double(std::rand()%tm_range)/tm_range - 0.5;
      double val = last_val *corr + renov*(1-corr);
      //val = 1.0;
      values.insert({ii, val});
      last_val = val;
    }

  constexpr long N = 1000;
  printf("\n");
  printf("tm,ma,mmv,mm,sema,semav,tema,lema,iema\n");
  MovingAverage ma(N);
  MovingMeanVariance mmv(N);
  MovingMean<double, N> mm;
  MovingAutoCorr<double> mac(N, 3);
  EMAAutoCorr<double> emac(N, 3);
  SimpleEMA<double> sema(halflife);
  SimpleEMAV<double> semav(halflife);
  TimeEMA<double, long> tema(halflife);
  LevelEMA<double, long> lema(halflife);
  ImpulseEMA<double, long> iema(halflife);

  for (long tm=0; tm<tm_range; tm++) {
    auto val_iter = values.find(tm);
    if (val_iter != values.end()) {
      double val = val_iter->second;
      printf("update tm: %ld, val %f", tm, val);
      ma.update(val);
      mmv.update(val);
      mm.update(val);
      mac.update(val);
      emac.update(val);
      sema.update(val);
      semav.update(val);
      tema.update(val, tm);
      lema.update(val, tm);
      iema.update(val, tm);
    } else {
      continue;
    }
    printf("%ld,",tm);
    printf("%f,",ma.value());
    printf("%f,",mmv.value());
    printf("%f,",mm.value());
    //printf("%f,",mac.correlations()[4]);
    auto corr = emac.correlations();
    for (auto cr : corr)
      std::cout<<" cr: "<<cr;
    std::cout<<std::endl;
    printf("%f,",sema.value());
    printf("%f,",semav.value());
    printf("%f,",tema.value());
    printf("%f,",lema.value(tm));
    printf("%f\n",iema.value(tm));


    //ema.update(v, v*0.1);
    //printf("input %f, average %f\n", v, ema.value());
  }
}

void CovarTest() {
  //MovingCovariance mc(100);
  SimpleEMAC mc(300);
  double rou = 0.1;
  printf("tm,va,cv,vb,cr,rou\n");
  for (int ii = 0; ii<1000; ++ii) {
    double xx = (std::rand()%1000)/1000.;
    double yy = (std::rand()%1000)/1000. * (1- rou)  + rou * xx;
    mc.update(xx, yy);
    double va=mc.variance_a();
    double vb=mc.variance_b();
    double cv=mc.covariance();
    double cr=cv/std::sqrt(va*vb);
    printf("%d,%f,%f%f,%f,%f,%f,%f\n", ii, xx,yy,va,cv,vb,cr,rou);
  }
}

void MultiCovarTest() {
  int len=std::rand()%100 + 100;
  //MultiMovingCovar<double> mmc(len, 3);
  MultiEMAC<double> mmc(len, 3);
  double rou1 = 0.1;
  double rou2 = 0.1;
  printf("tm,v0,v1,v2,c01,c22,c12,cr01,cr02,cr02\n");
  for (int ii = 0; ii<1000; ++ii) {
    double xx = (std::rand()%1000)/1000.;
    double yy = (std::rand()%1000)/1000. * (1- rou1)  + rou1 * xx;
    double zz = (std::rand()%1000)/1000. * (1- rou2)  + rou2 * xx;
    mmc.update({xx, yy, zz});
    std::valarray<double> means = mmc.means();
    std::valarray<double> covar = mmc.covariance();
    double v0 = covar[0];
    double v1 = covar[4];
    double v2 = covar[8];
    double c01 = covar[3];
    double c02 = covar[6];
    double c12 = covar[7];
    double cr01=c01/std::sqrt(v0*v1);
    double cr02=c02/std::sqrt(v0*v2);
    double cr12=c12/std::sqrt(v1*v2);
    printf("%d,%f,%f,%f,%f,%f,%f,%f ,%f, %f, %f,  %f,  %f\n",
            ii,means[0],means[1],means[2],v0,v1,v2,c01,c02,c12,cr01,cr02,cr12);
  }
}


void TestValArray() {
  size_t ran = std::rand()%100;
  //std::valarray<double> va(0.0, ran);
  std::valarray<int> va{0,  1,  2, 3, 4, 5, 6,
                       10, 11,12,13,14,15,16,
                       20, 21,22,23,24,25,26,
                       30, 31,32,33,34,35,36};
  for (auto v:va) {printf("%d,",v);}
  printf("\n");
  std::slice_array<int> sa = va[std::slice(1,4,7)];
  std::valarray<int> result{1,2,3,4};
  //result -= sa;
  sa -= result;
  for (auto v : std::valarray<int>(sa)) { printf("%d,",v);}
  printf("\n");
  /*
  std::valarray<int> sa = cross_prod(va);
  size_t sz=va.size();
  for (size_t ii = 0; ii<sz*sz; ++ii) {
    printf("%d,", sa[ii]);
    if (ii%sz==0) {printf("\n");}
  }*/

  for (auto v:va) {printf("%d,",v);}
  printf("\n");
  sa = result;
  for (auto v:va) {printf("%d,",v);}
  //for (size_t ii=0;  ii<7; ii++) { printf("%d,", sa[ii]); }


}

int main() {
  //user();
  //concurrent();
  //CompileTimeConditional();
  StatsOutput();
  //MultiCovarTest();
  //TestValArray();
  return 0;
}
