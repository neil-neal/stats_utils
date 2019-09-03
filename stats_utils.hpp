#pragma once

/*
 *  Efficient implementation of statistical utility functions, such as moving
 * average, standard dieviation, (co)variance etc
 */


#include <cmath>
#include <type_traits>
#include <tuple>
#include <valarray>
#include <vector>
#include <iostream>


/*
 *  Equal weight moving average in rolling window of size N samples
 */
template <typename valT = double > class MovingAverage {
public:
  using retT = decltype(valT()*double(1.0));

  MovingAverage(long N) : N_{N}, invertN_{1.0/N} {
    data_ = new valT[N];
  }
  MovingAverage(MovingAverage & rhs) = delete;

  ~MovingAverage() {    delete[] data_;  }

  retT update(valT val) {
    valT dff = val - data_[idx_];
    data_[idx_] = val;
    idx_ = ++idx_%N_;
    sum_ += dff;
    divisor_ = ++n_ < N_ ? 1.0/n_ : invertN_;
    return average_ = sum_*divisor_;
  }

  retT value() const {
    return average_;
  }

protected:
  valT* data_;
  double divisor_{1.0};
  valT sum_{0};
  long idx_{0};

private:
  long N_{0};
  long n_{0};
  retT invertN_{1.0};
  retT average_;

};



/*
 *  Equal weight moving average and variance in rolling window of size N samples
 */
template<typename valT = double >
class MovingMeanVariance : public MovingAverage<valT> {
  using retT = typename MovingAverage<valT>::retT;
public:
  MovingMeanVariance(long N) : MovingAverage<valT>(N) {}

  void update(valT val) {
    sum_of_squares_ += val*val - this->data_[this->idx_]*this->data_[this->idx_];
    MovingAverage<valT>::update(val);
    variance_ = (sum_of_squares_ -
                 this->sum_*this->sum_*this->divisor_)*this->divisor_;
  }

  retT variance() const {
    return variance_;
  }

private:
  valT sum_of_squares_{};
  retT variance_;
};



/*
 * Covariance (and means) of two variables series of rolling window size N
 */
template<typename valT = double > class MovingCovariance {
  using retT = typename MovingAverage<valT>::retT;
public:
  MovingCovariance(long N) : N_{N}, invertN_{1.0/N} {
    data_a_ = new valT[N];
    data_b_ = new valT[N];
  }

  ~MovingCovariance() {  delete[] data_a_;  delete[] data_b_; }

  void update(valT a, valT b) {
    valT old_a = data_a_[idx_];
    valT old_b = data_b_[idx_];
    sum_a_ += a - old_a;
    sum_b_ += b - old_b;
    sum_of_sqrs_a_ += a*a - old_a * old_a;
    sum_of_sqrs_b_ += b*b - old_b * old_b;
    sum_of_axb_ += a*b - old_a * old_b;
    data_a_[idx_] = a;
    data_b_[idx_] = b;
    idx_ = ++idx_%N_;

    divisor_ = ++n_ < N_ ? 1.0/n_ : invertN_;
    average_a_ = sum_a_*divisor_;
    average_b_ = sum_b_*divisor_;
    variance_a_ = (sum_of_sqrs_a_ - sum_a_ * sum_a_ * divisor_) * divisor_;
    variance_b_ = (sum_of_sqrs_b_ - sum_b_ * sum_b_ * divisor_) * divisor_;
    covar_ab_ = sum_of_axb_ * divisor_ - average_a_ * average_b_;
  }

  retT average_a() const { return average_a_; }
  retT average_b() const { return average_b_; }
  std::pair<retT, retT> averages() const {
    return std::make_pair(average_a_, average_b_);
  }
  retT variance_a() const { return variance_a_; }
  retT variance_b() const { return variance_b_; }
  retT covariance() const { return covar_ab_; }
  std::tuple<retT, retT, retT> covariances() const {
    return std::make_tuple(variance_a_, covar_ab_, variance_b_);
  }


private:
  valT* data_a_;  valT* data_b_;
  double divisor_{1.0};
  valT sum_a_{0};   valT sum_b_{0};
  valT sum_of_sqrs_a_{0};   valT sum_of_sqrs_b_{0};
  valT sum_of_axb_{0};
  long idx_{0};
  long N_{0};
  long n_{0};
  retT invertN_{1.0};
  retT average_a_;   retT average_b_;
  retT variance_a_, covar_ab_, variance_b_;
};



/*
 *  Moving auto-correlation with maximum lag maxLag
 */
template <typename valT> class MovingAutoCorr {
public:
  MovingAutoCorr(size_t length, size_t maxlag)
    : N_{length}, N_plus_lag_{length + maxlag}
    , invertN_{1.0/N_}
    , max_lag_{maxlag}
    , idxs_(max_lag_)
    , data_(valT(), N_ + max_lag_)
    , sums_(valT(), max_lag_)
    , sums_of_sqrs_(valT(), max_lag_)
    , sums_of_prod_(valT(), max_lag_) {
    for (int ii = 0; ii<max_lag_; ++ii) {
      idxs_[ii] = ii;
    }
  }

  void update(valT val) {
    // (idxs_ + max_lag_)%N_plus_lag_ == (idxs_ - N_)%N_plus_lag_
    //std::cout<<"MAC updated "<<val<<std::endl;
    std::valarray<valT> old_vals = data_[(idxs_ + max_lag_)%N_plus_lag_];
    data_[idxs_[max_lag_ -1]] = val;
    std::valarray<valT> new_vals = data_[idxs_];
    sums_ += new_vals - old_vals;
    sums_of_sqrs_ += new_vals * new_vals - old_vals * old_vals;
    sums_of_prod_ += val * new_vals - old_vals[max_lag_-1] * old_vals;
    (idxs_ += 1) %= N_plus_lag_;
    ++n_;
    need_recalc_ = true;
  }

  std::valarray<valT> const & means() {
    if (need_recalc_)
      update_output_();
    return means_;
  }

  std::valarray<valT> const & covariances() {
    if (need_recalc_)
      update_output_();
    return covars_;
  }

  std::valarray<valT> const & correlations() {
    if (need_recalc_)
      update_output_();
    return correls_;
  }

  std::valarray<valT> const & values() {
    return correlations();
  }


private:
  /* Member functions
   */
  void update_output_() {
    double divisor = n_ < N_ ? 1.0/n_ : invertN_;
    means_ = sums_ * divisor;
    variances_ = sums_of_sqrs_*divisor - means_ * means_;
    covars_ = sums_of_prod_*divisor - means_ * means_[max_lag_ - 1];
    correls_ = covars_/std::sqrt(variances_ * variances_[max_lag_ -1]);
    need_recalc_ = false;
  }

  /* Member variables
   */
  size_t N_{0};
  size_t N_plus_lag_{0};
  size_t n_{0};
  double invertN_{};
  size_t max_lag_;
  std::valarray<size_t> idxs_;
  std::valarray<valT> data_;
  std::valarray<valT> sums_;
  std::valarray<valT> sums_of_prod_;
  std::valarray<valT> sums_of_sqrs_;
  bool need_recalc_{false};
  std::valarray<valT> means_;
  std::valarray<valT> variances_;
  std::valarray<valT> covars_;
  std::valarray<valT> correls_;
};


/*
 *  Exponential moving auto correlation
 */
template <typename valT> class EMAAutoCorr {
public:
  EMAAutoCorr(double halflife, size_t maxlag)
    : decay_{1.0/std::pow(2.0, 1.0/halflife)}
    , max_lag_{maxlag}, last_{max_lag_ -1}
    , data_(valT(), max_lag_)
    , weighed_sums_(valT(), max_lag_)
    , idxs_(max_lag_)
    , weighed_sqr_sums_(valT(), max_lag_)
    , weighed_prod_sums_(valT(), max_lag_)
    , total_weights_(0.0, max_lag_)
    , variances_(valT(), max_lag_)
    , covars_(valT(), max_lag_)  {
    for (int ii = 0; ii<max_lag_; ++ii)
      idxs_[ii] = ii;
  }

  void update(valT val) {
    size_t current_idx = idxs_[last_];
    size_t last_idx = (current_idx + max_lag_ - 1) % max_lag_;
    data_[current_idx] = val;
    weighed_sums_[current_idx] = weighed_sums_[last_idx] * decay_;
    weighed_sums_[current_idx] += val;
    weighed_sqr_sums_[current_idx] = weighed_sqr_sums_[last_idx] * decay_;
    weighed_sqr_sums_[current_idx] += val*val;
    weighed_prod_sums_ *= decay_;
    std::valarray<valT> prod = data_[idxs_];
    prod *= val;  // valaray<> support only *= operation, not binary op *
    weighed_prod_sums_ += prod;
    total_weights_[current_idx] = total_weights_[last_idx] * decay_;
    total_weights_[current_idx] += 1;
    (idxs_ += 1) %= max_lag_;
    need_recalc_ = true;
  }


  std::valarray<valT> const & means() {
    if (need_recalc_)
      update_output_();
    return means_;
  }

  std::valarray<valT> const & covariances() {
    if (need_recalc_)
      update_output_();
    return covars_;
  }

  std::valarray<valT> const & correlations() {
    if (need_recalc_)
      update_output_();
    return correls_;
  }

  std::valarray<valT> const & values() {
    return correlations();
  }


private:
  /* Member functions */
  void update_output_() {
    size_t current_idx = idxs_[last_];
    means_ = weighed_sums_/ total_weights_;
    variances_ = weighed_sqr_sums_ / total_weights_ - means_ * means_;
    covars_ = weighed_prod_sums_/total_weights_ - means_ * means_[current_idx];
    correls_ = covars_/std::sqrt(variances_ * variances_[current_idx]);
    need_recalc_ = false;
  }

private:
  /* Data members */
  double decay_;
  size_t max_lag_;
  size_t last_;
  std::valarray<valT> data_;
  std::valarray<valT> weighed_sums_;
  std::valarray<size_t> idxs_;
  std::valarray<valT> weighed_sqr_sums_;
  std::valarray<valT> weighed_prod_sums_;
  std::valarray<double> total_weights_;
  std::valarray<valT> variances_;
  std::valarray<valT> covars_;

  bool need_recalc_{false};
  std::valarray<valT> means_;
  std::valarray<valT> correls_;
};


/*
 * cross product of vector; output a N x N lower square matrix
 * stored in a valarray of size N^2
 */
template<typename valT>
std::valarray<valT> cross_prod(std::valarray<valT> const & a) {
  const size_t sz = a.size();
  std::valarray<valT> ret_val(valT(), sz*sz);

  for (size_t ii = 0; ii<sz; ++ii) {
    for (size_t jj = 0; jj<=ii; ++jj) {
      ret_val[ii*sz + jj] = a[ii]*a[jj];
    }
  }
  return ret_val;
}

/*
 * Covariance (and means) of M variables series of rolling window size N
 */
template<typename valT = double > class MultiMovingCovar {
public:
  MultiMovingCovar(long N, int M) : N_{N}, M_{M}, invertN_{1.0/N} {
    // N rows by M columns
    data_ = std::valarray<valT>(valT(), N_*M_);
    sums_ = std::valarray<valT>(valT(), M_);
    prod_sums_ = std::valarray<valT>(valT(), M_*M_);
  }

  void update(std::valarray<valT> const & vals) {
    std::slice_array<valT> old_vals = data_[std::slice(idx_*M_, M_, 1)];
    (sums_ += vals) -= old_vals;
    (prod_sums_ += cross_prod(vals))-= cross_prod(std::valarray<valT>(old_vals));
    old_vals = vals;
    idx_ = ++idx_%N_;

    double divisor = ++n_ < N_ ? 1.0/n_ : invertN_;
    means_ = sums_ * divisor;
    covariance_ = (prod_sums_ - cross_prod(sums_) * divisor) * divisor;
  }

  std::valarray<valT> const & means() const { return means_;}
  std::valarray<valT> const & covariance() const { return covariance_;}

private:
  std::valarray<valT> data_;
  std::valarray<valT> sums_;
  std::valarray<valT> prod_sums_;
  long N_{0};
  long n_{0};
  long idx_{0};
  double invertN_;
  int M_{0};
  std::valarray<valT> means_;
  std::valarray<valT> covariance_;
};


/*
 * Exponentially weighted covariance matrix of M (>=2) variables
 */
template<typename valT = double> class MultiEMAC {
public:
  MultiEMAC(double hl, int M ) : decay_{1.0/std::pow(2.0, 1.0/hl)}, M_{M} {
    weighted_sums_ = std::valarray<valT>(valT(), M_);
    weighted_prod_sums_ = std::valarray<valT>(valT(), M_*M_);
  }

  void update(std::valarray<valT> const & vals) {
    (weighted_sums_ *= decay_) += vals;
    (weighted_prod_sums_ *= decay_) += cross_prod(vals);
    (weight_sum_ *= decay_)++;

    double inv_wt = 1./weight_sum_;
    means_ = weighted_sums_ * inv_wt;
    covariance_ = (weighted_prod_sums_ - cross_prod(weighted_sums_)*inv_wt)
                                                  * inv_wt;
  }

  std::valarray<valT> const & means() const {  return means_;  }
  std::valarray<valT> const & covariance() const {  return covariance_; }

private:
  double decay_{0.5}; // ~ halflife = 1
  int M_;
  std::valarray<valT> weighted_sums_;
  std::valarray<valT> weighted_prod_sums_;
  double weight_sum_{0.0};
  std::valarray<valT> means_;
  std::valarray<valT> covariance_;
};


/*
 *  Equal weight moving average in rolling window of COMPILE TIME fixed size
 *  N samples
 */
template <typename valT = double, auto N = 1024 > class MovingMean {
public:
  static constexpr long largeN = 1024*1024/sizeof(valT);
  using dtype =
      typename std::conditional<N<largeN, std::array<valT, N>, valT*>::type;
  using sizeT = decltype(N);
  using retT = decltype(valT()/double(N));
  // using heap in place of stack to store data when N >= largeN

  //SFINAE
  template<long M, typename std::enable_if< ! (M>=largeN) >::type * = nullptr>
    void init_dat(){
    data_ = std::array<valT, M>{{valT()}};
    //std::cout<<"data_ on stack\n"<<std::endl;
    }

  template<long M, typename std::enable_if< M>=largeN >::type * = nullptr>
    void init_dat(){
    data_ = new valT[M];
    //std::cout<<"data_ on heap\n"<<std::endl;
  }

  MovingMean() {
    init_dat<N>();
  }

  decltype(auto) update(valT val) {
    valT dff = val - data_[idx_];
    data_[idx_] = val;
    idx_ = ++idx_%N;
    sum_ += dff;
    return average_ = sum_*(++n_ < N ? 1.0/n_ : invertN_);
  }

  decltype(auto) value() const {
    return average_;
  }


private:
  dtype data_;
  sizeT n_{0};
  sizeT idx_{0};
  valT sum_{0};
  retT invertN_{1.0/N};
  retT average_;
};



/*
 * Exponentially weighted moving average, where weights of earlier samples
 *  decay same proportion with each new sample update.
 */
template<typename valT = double > class SimpleEMA {
public:
  using retT = decltype(valT()*double(1.0));
  SimpleEMA(double hl ) : decay_{1.0/std::pow(2.0, 1.0/hl)} {}

  void update(valT val) {
    (weighted_vals_sum_ *= decay_) += val;
    (weight_sum_ *= decay_)++;
    //(value_ *= decay_) += (1-decay_)*val;
    //printf("%f, %f, %f\n", val,weighted_val_sum_, weight_sum_);
  }

  retT value() const {
    return weighted_vals_sum_/weight_sum_;
    //return value_;
  }

protected:
  double decay_{0.5}; // ~ halflife = 1
  retT weighted_vals_sum_{};
  double weight_sum_{0.0};
  //retT value_{};
};


/*
 * Exponentially weighted moving average and variance, where weights of
 * earlier samples decay same proportion with each new sample update.
 */
template<typename valT = double > class SimpleEMAV : public SimpleEMA<valT> {
  using retT = typename SimpleEMA<valT>::retT;
public:
  SimpleEMAV(double hl ) : SimpleEMA<valT>(hl) {}

  void update(valT val) {
    (weighted_sqrs_sum_ *= this->decay_) += val * val;
    SimpleEMA<valT>::update(val);
  }

  retT variance() const {
    return (weighted_sqrs_sum_ -
            (this->weighted_vals_sum_ * this->weighted_vals_sum_)
              /this->weight_sum_)/this->weight_sum_;
  }

private:
  retT weighted_sqrs_sum_{};
};


/*
 * Exponentially weighted (co)variance, where weights of
 * earlier samples decay same proportion with each new sample update.
 */
template<typename valT = double> class SimpleEMAC {
public:
  using retT = decltype(valT()*double(1.0));
  SimpleEMAC(double hl ) : decay_{1.0/std::pow(2.0, 1.0/hl)} {}

  void update(valT a, valT b) {
    (weighted_a_sum_ *= decay_) += a;
    (weighted_aa_sum_ *= decay_) += a*a;
    (weighted_b_sum_ *= decay_) += b;
    (weighted_bb_sum_ *= decay_) += b*b;
    (weighted_axb_sum_ *= decay_) += a*b;
    (weight_sum_ *= decay_)++;

    double inv_wt = 1./weight_sum_;
    mean_a_ = weighted_a_sum_ * inv_wt;
    mean_b_ = weighted_b_sum_ * inv_wt;
    var_a_ = (weighted_aa_sum_ - weighted_a_sum_ * weighted_a_sum_ * inv_wt)
                  *inv_wt;
    var_b_ = (weighted_bb_sum_ - weighted_b_sum_ * weighted_b_sum_ * inv_wt)
                  *inv_wt;
    covar_ = (weighted_axb_sum_ - weighted_a_sum_ * weighted_b_sum_ * inv_wt)
                  *inv_wt;
    //printf("%f, %f, %f\n", val,weighted_val_sum_, weight_sum_);
  }

  retT mean_a() const {  return mean_a_;  }
  retT mean_b() const {  return mean_b_;  }
  retT variance_a() const {  return var_a_;  }
  retT variance_b() const {  return var_b_;  }
  retT covariance() const {  return covar_;  }

protected:
  double decay_{0.5}; // ~ halflife = 1
  retT weighted_a_sum_{}; retT weighted_b_sum_{};
  retT weighted_aa_sum_{}; retT weighted_bb_sum_{}; retT weighted_axb_sum_{};
  double weight_sum_{0.0};
  retT mean_a_{}; retT mean_b_{}; retT var_a_{}; retT var_b_{}; retT covar_{};
};


/*
 *  Exponentially weighted moving average, where weights are proportion
 * to negative exponential of time passed since the samples were added
 */
template<typename valT = double, typename timeT = double > class TimeEMA {
  using retT = decltype(valT()*double(1.0));
 public:
  TimeEMA(timeT hl ) : adbmal_{std::log(2.0)/hl} {}
  void update(valT val, timeT tm) {
    double decay = std::exp((last_update_tm_ - tm)*adbmal_);
    (numerator_ *= decay) += val;
    (denominator_ *= decay)++;
    last_update_tm_ = tm;
    //printf("%f, %f, %f\n", val, numerator_, denominator_);
  }

  retT value() const {
    return numerator_/denominator_;
  }

private:
  timeT last_update_tm_{};
  decltype(double(1.0)/timeT()) adbmal_;  // inverse of lambda
  retT numerator_{0.0};
  double denominator_{0.0};
};


/*
 * Internal vValue decays toward zero as time goes by. When updated, new
 * value is added to the internal value. The value-time graph looks like
 * sudden increases (impulses) followed by exponential decay toward zero.
 */
template<typename valT = double, typename timeT = double > class ImpulseEMA {
 public:
  ImpulseEMA(timeT hl ) : adbmal_{std::log(2.0)/hl} {}

  valT update(valT impulse, timeT tm) {
    value_ = value(tm) + impulse;
    last_update_tm_ = tm;
    //printf("%f, %f, %f\n", value_, latest_level_, last_update_tm_);
    return value_;
  }

  valT value(timeT tm) const {
    double decay = std::exp((last_update_tm_ - tm)*adbmal_);
    return value_ * decay;
  }

  valT value() const {
    return value_;
  }

private:
  timeT last_update_tm_{};
  decltype(double(1.0)/timeT()) adbmal_;  // inverse of lambda
  valT value_{};
};



/*
 * At each sampling/update, the EMA value asymptotes toward the
 * the latest update value/level exponentially.
 * Unlike others, the EMA internal value is not affected by
 * sampling frequency
 */
template<typename valT = double, typename timeT = double> class LevelEMA {
public:
  LevelEMA(timeT hl, valT init_val = valT{}, timeT start_tm = timeT{} ) :
    adbmal_{std::log(2.0)/hl}, value_{init_val}, last_update_tm_{start_tm} {}

  valT update(valT lvl, timeT tm) {
    value_ = value(tm);
    latest_level_ = lvl;
    last_update_tm_ = tm;
    //printf("%f, %f, %f\n", value_, latest_level_, last_update_tm_);
    return value();
  }

  valT value(timeT tm) const {
    double decay = std::exp((last_update_tm_ - tm)*adbmal_);
    return latest_level_ + (value_ - latest_level_)*decay;
  }

  valT value() const {
    return value_;
  }

private:
  timeT last_update_tm_{};
  decltype(double(1.0)/timeT()) adbmal_;  // inverse of lambda
  valT latest_level_;
  valT value_;
};





