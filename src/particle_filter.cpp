/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Number of particles
  num_particles = 200;

  default_random_engine gen;
  // Create normal distribution to generate x, y and theta values for Particles with Gaussian noise.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  cout << "ParticleFilter::init num_particles:" << num_particles << endl;
  cout << " x:" << x << " y:" << y << " theta:" << theta;
  cout << " std_dev: " << std[0] << " " << std[1] << " " << std[2] << endl;

  // Initialize all particles
  for (auto i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(p.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // TODO: Add measurements to each particle and add random Gaussian noise.

  default_random_engine gen;
  // Create normal distributions for velocity and yaw rate.
  normal_distribution<double> dist_vel(velocity, std_pos[0]);
  normal_distribution<double> dist_yawd(yaw_rate, std_pos[1]);

  cout << "DEBUG: ParticleFilter::prediction: delta_t:" << delta_t << " velocity:" << velocity;
  cout << " std_pos:" << std_pos[0] << std_pos[1] << " yaw_rate:" << yaw_rate << endl;

  for (auto& p : particles) {
    const auto v = dist_vel(gen);
    const auto yawd = dist_yawd(gen);
    if (fabs(yawd) > 0.001) {
      const double v_over_yawd = v / yawd;
      p.x += v_over_yawd * (sin(p.theta + yawd * delta_t) - sin(p.theta));
      p.y += v_over_yawd * (cos(p.theta) - cos(p.theta + yawd * delta_t));
      p.theta += (yawd * delta_t); // TODO: Check if this needs to be normalized.
    } else {
      // Almost 0 yaw rate. Assume the car is moving on a straight line.
      p.x += v * delta_t * cos(p.theta);
      p.y += v * delta_t * sin(p.theta);
    }
  }

  return;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  // Make sure we have atleast one predicted landmark
  if (predicted.empty()) {
    cout << "ParticleFilter::dataAssociation Empty predicted vector\n";
    return;
  }

  for (auto& obs : observations) {
    double min_distance = numeric_limits<double>::max();  // DOUBLE_MAX
    for (const auto& pre : predicted) {
      // Find the distance between observed landmark and landmark on the map
      const auto distance = dist(obs.x, obs.y, pre.x, pre.y);
      if (distance < min_distance) {
        min_distance = distance;
        obs.id = pre.id;
      }
    }
  }

  return;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  cout << "UpdateWeights: sensor_range:" << sensor_range;
  cout << " std_landmark" << std_landmark[0] << std_landmark[1];
  cout << " observations: " << observations.size();
  cout << " Landmarks: " << map_landmarks.landmark_list.size() << endl;

  int max_predictions = 0;

  // Clear weights
  weights.clear();

  // For each particle
  for (auto& p : particles) {

    // Reset the weight
    p.weight = 1.0;

    // Convert vehicle's observations in to map co-ordinate system
    // using Homogeneous transformation
    // xm = px + cos(ptheta)*x_obs - sin(ptheta)*y_obs
    // ym = py + sin(ptheta)*x_obs + cos(ptheta)*y_obs
    vector<LandmarkObs> tx_observations;
    for (const auto& obs : observations) {
      auto xm = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
      auto ym = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;
      LandmarkObs tx_obs;
      tx_obs.x = xm;
      tx_obs.y = ym;
      tx_obs.id = obs.id;
      tx_observations.push_back(tx_obs);
    }

    // Find a list of landmarks within the sensor range.
    vector<LandmarkObs> predictions;
    for (const auto& map_lm : map_landmarks.landmark_list) {
      // Check distance from the particle
      const auto distance = dist(p.x, p.y, map_lm.x_f, map_lm.y_f);
      if (distance < sensor_range) {
        LandmarkObs pre;
        pre.x = map_lm.x_f;
        pre.y = map_lm.y_f;
        pre.id = map_lm.id_i;
        predictions.push_back(pre);
      }
    }

    if (predictions.size() > max_predictions)
      max_predictions = predictions.size();

    // Associate transformed observations with the predicted landmarks
    dataAssociation(predictions, tx_observations);

    // For each transformed observation
    for (const auto& obs : tx_observations) {
      // Find associatd prediction
      for (const auto& pre : predictions) {
        if (pre.id == obs.id) {
          // Calculate the weight for this observation using
          // Multi-variate Gaussian distribution
          const auto sigma_x = std_landmark[0];
          const auto sigma_y = std_landmark[1];

          const auto gaus_norm = 1 / (2 * M_PI * sigma_x * sigma_y);
          const auto exp_x = ((obs.x - pre.x) * (obs.x - pre.x))
              / (2 * sigma_x * sigma_x);
          const auto exp_y = ((obs.y - pre.y) * (obs.y - pre.y))
              / (2 * sigma_y * sigma_y);
          const auto exp_x_y = -(exp_x + exp_y);
          const auto w_obs = gaus_norm * exp(exp_x_y);

          // Multiply the observation's weight to particle's weigth
          p.weight *= w_obs;

          break;
        }
      }
    }

    // Push particle's weight to the filter's list
    weights.push_back(p.weight);
  }

  cout << "DEBUG: Update: max predictions:" << max_predictions << endl;

  return;
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  if (weights.size() != particles.size()) {
    cout << "ERROR: ParticleFilter::resample(): weights.size() != particles.size()\n";
    return;
  }

  // copy current particles in to a temp vector
  vector<Particle> old_particles = particles;

  // clear particles
  particles.clear();

  // Create an uniform integer distribution for indices.
  const auto num_weights = weights.size();
  const int last_idx = num_weights - 1;
  uniform_int_distribution<int> dist_index(0, last_idx);

  // Create an uniform real distribution for weights.
  const auto w_max = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> dist_beta(0, 2 * w_max);

  default_random_engine gen;

  cout << "DEBUG: ParticleFilter::resample(): w_max: " << w_max << " num_weights:"<< num_weights << endl;

  // Start with a random index
  auto idx = dist_index(gen);
  double beta = 0;

  for (auto i = 0u; i < num_weights; i++) {
    beta += dist_beta(gen);
    while (beta > weights[idx]) {
      beta -= weights[idx];
      // increment with a wrap around.
      idx = (idx == last_idx) ? 0 : idx+1;
    }

    // Add the new weight to list.
    particles.push_back(old_particles[idx]);
  }

  if (particles.empty()) {
    cout << "ParticleFilter::resample No particles left after resample\n";
  }

  return;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
