/**
 * Framework for 2048 & 2048-like Games (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <math.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include "action.h"
#include "board.h"
#include "weight.h"
#define FEATURES_NUM 8

class agent {
   public:
    agent(const std::string& args = "") {
        std::stringstream ss("name=unknown role=unknown " + args);
        for (std::string pair; ss >> pair;) {
            std::string key = pair.substr(0, pair.find('='));
            std::string value = pair.substr(pair.find('=') + 1);
            meta[key] = {value};
        }
    }
    virtual ~agent() {}
    virtual void open_episode(const std::string& flag = "") {}
    virtual void close_episode(const std::string& flag = "") {}
    virtual action take_action(const board& b) { return action(); }
    virtual bool check_for_win(const board& b) { return false; }

   public:
    virtual std::string property(const std::string& key) const { return meta.at(key); }
    virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = {msg.substr(msg.find('=') + 1)}; }
    virtual std::string name() const { return property("name"); }
    virtual std::string role() const { return property("role"); }

   protected:
    typedef std::string key;
    struct value {
        std::string value;
        operator std::string() const { return value; }
        template <typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
        operator numeric() const { return numeric(std::stod(value)); }
    };
    std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
   public:
    random_agent(const std::string& args = "") : agent(args) {
        if (meta.find("seed") != meta.end())
            engine.seed(int(meta["seed"]));
    }
    virtual ~random_agent() {}

   protected:
    std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
   public:
    weight_agent(const std::string& args = "") : agent(args), alpha(0) {
        if (meta.find("init") != meta.end())
            init_weights(meta["init"]);
        if (meta.find("load") != meta.end())
            load_weights(meta["load"]);
        if (meta.find("alpha") != meta.end())
            alpha = float(meta["alpha"]);
    }
    virtual ~weight_agent() {
        if (meta.find("save") != meta.end())
            save_weights(meta["save"]);
    }

    const std::vector<std::vector<int>> features = {
        {0, 1, 2, 3, 4},
        {4, 5, 6, 7, 8},
        {5, 6, 7, 9, 10},
        {9, 10, 11, 13, 14},

        {0, 4, 8, 12, 13},
        {1, 5, 9, 13},
        {1, 2, 5, 6, 9},
        {2, 3, 6, 7, 10},

        {11, 12, 13, 14, 15},
        {7, 8, 9, 10, 11, 12},
        {5, 6, 8, 9, 10},
        {1, 2, 4, 5, 6},

        {2, 3, 7, 11, 15},
        {1, 2, 6, 10, 14},
        {1, 2, 5, 6, 10,},
        {0, 1, 4, 5, 9}
        // {0, 4, 8, 9, 12, 13},
        // {1, 5, 9, 10, 13, 14},
        // {1, 2, 5, 6, 9, 10},
        // {2, 3, 6, 7, 10, 11},

        // {0, 1, 2, 3, 4, 5},
        // {4, 5, 6, 7, 8, 9},
        // {5, 6, 7, 9, 10 , 11},
        // {9, 10, 11, 13, 14, 15},

        // {1, 2, 5, 6, 10, 14},
        // {2, 3, 6, 7, 11, 15},
        // {4, 5, 8, 9, 12, 13},
        // {5, 6, 9, 10, 13, 14},

        // {6, 7, 8, 9, 10, 11},
        // {10, 11, 12, 13, 14, 15},
        // {0, 1, 2, 4, 5, 6},
        // {4, 5, 6, 8, 9, 10},

    };
    const std::vector<std::vector<int>> v_tile_features = {
        // {0, 1, 2, 3},
        // {4, 5, 6, 7},
        // {8, 9, 10, 11},
        // {12, 13, 14, 15},
        // {0, 4, 8, 12},
        // {1, 5, 9, 13},
        // {2, 6, 10, 14},
        // {3, 7, 11, 15}
        {0, 4, 8, 9, 12, 13},
        {1, 5, 9, 10, 13, 14},
        {1, 2, 5, 6, 9, 10},
        {2, 3, 6, 7, 10, 11},
    };
    int myPow(int x, unsigned int p) {
        if (p == 0) return 1;
        if (p == 1) return x;

        int tmp = myPow(x, p / 2);
        if (p % 2 == 0)
            return tmp * tmp;
        else
            return x * tmp * tmp;
    }
    int extract_feature(const board& after, std::vector<int> feature) {
        int idx = 0;
        // for (long unsigned int i = 0; i < feature.size(); i++) {
        //     idx += after(feature[i]) * myPow(25, feature.size() - i - 1);
        // }
        for (long unsigned int i = 0; i < feature.size(); i++) {
            idx *= 25;
            idx += after(feature[i]);
        }
        return idx;
    };

    float estimate_value(const board& after) {
        float value = 0;
        for (int i = 0; i < FEATURES_NUM; i++) {
            value += net[i][extract_feature(after, features[i])];
        }
        return value;
    };

    typedef struct step {
        board state;
        board::reward reward;
        bool terminated;

    } Step;

    void td_0(Step last, const board& next) {
        float current = estimate_value(last.state);
        float target = last.terminated ? last.reward : estimate_value(next) + last.reward;
        float error = target - current;
        for (int i = 0; i < FEATURES_NUM; i++) {
            net[i][extract_feature(last.state, features[i])] += alpha * error;
        }
    };

    virtual action take_action(const board& before) {
        if (!last_step.empty()) {
            td_0(last_step.back(), before);
            last_step.pop_back();
        }

        int best_op = -1;
        int best_reward = -1;
        float best_value = -100000;

        for (int op : {0, 1, 2, 3}) {
            board after = board(before);
            board::reward reward = after.slide(op);
            if (reward == -1) continue;
            float value = estimate_value(after);
            if (reward + value >= best_reward + best_value) {
                best_op = op;
                best_reward = reward;
                best_value = value;
            }
        }
        Step last = {before, best_reward, best_op == -1};
        last_step.emplace_back(last);
        return action::slide(best_op);
    };
    virtual void open_episode(const std::string& flag = "") {
        last_step.clear();
    };
    virtual void close_episode(const std::string& flag = ""){};

   protected:
    virtual void init_weights(const std::string& info) {
        for (auto feature : features) {
            net.emplace_back(myPow(25, feature.size()));
        }
        //		net.emplace_back(65536); // create an empty weight table with size 65536
        //		net.emplace_back(65536); // create an empty weight table with size 65536
    }
    virtual void load_weights(const std::string& path) {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in.is_open()) std::exit(-1);
        uint32_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        net.resize(size);
        for (weight& w : net) in >> w;
        in.close();
    }
    virtual void save_weights(const std::string& path) {
        std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!out.is_open()) std::exit(-1);
        uint32_t size = net.size();
        out.write(reinterpret_cast<char*>(&size), sizeof(size));
        for (weight& w : net) out << w;
        out.close();
    }

   protected:
    std::vector<weight> net;
    float alpha;
    std::vector<Step> last_step;
};
/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
   public:
    rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
                                           space({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
                                           popup(0, 9) {}

    virtual action take_action(const board& after) {
        std::shuffle(space.begin(), space.end(), engine);
        for (int pos : space) {
            if (after(pos) != 0) continue;
            board::cell tile = popup(engine) ? 1 : 2;
            return action::place(pos, tile);
        }
        return action();
    }

   private:
    std::array<int, 16> space;
    std::uniform_int_distribution<int> popup;
};

/**
 * dummy player
 * select a legal action randomly
 */
