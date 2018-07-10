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
  num_particles = 100;
  const char* env_var = getenv("NUM_PARTICLES");
  if (env_var != NULL) {
    num_particles = atoi(env_var);
  }
  cout << "Initalizing particle filter with " << num_particles << " particles\n";

  default_random_engine gen;
  // Create normal distribution to generate x, y and theta values for Particles with Gaussian noise.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

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
  // Create normal distribution to generate dx, dy and dtheta values for Particles with Gaussian noise.
  normal_distribution<double> dist_dx(0, std_pos[0]);
  normal_distribution<double> dist_dy(0, std_pos[1]);
  normal_distribution<double> dist_dtheta(0, std_pos[2]);

  bool constant_yaw = fabs(yaw_rate) < 0.001;
  const double v_over_yawd = velocity / yaw_rate;
  const double dt_yaw_rate = yaw_rate * delta_t;
  const double dt_velocity = velocity * delta_t;

  for (auto& p : particles) {
    if (constant_yaw) {
      // Assume the car is moving on a straight line.
      p.x += dt_velocity * cos(p.theta);
      p.y += dt_velocity * sin(p.theta);
    } else {
      p.x += v_over_yawd * (sin(p.theta + dt_yaw_rate) - sin(p.theta));
      p.y += v_over_yawd * (cos(p.theta) - cos(p.theta + dt_yaw_rate));
      p.theta += dt_yaw_rate;
    }

    // Add noise
    p.x += dist_dx(gen);
    p.y += dist_dy(gen);
    p.theta += dist_dtheta(gen);
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

  return;
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // copy current particles in to a temp vector
  vector<Particle> new_particles;

  discrete_distribution<int> dist_index(weights.begin(), weights.end());
  default_random_engine gen;
  for (auto i = 0u; i < weights.size(); i++) {
    // Add the new weight to list.
    new_particles.push_back(particles[dist_index(gen)]);
  }
  particles = new_particles;

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
