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

    // const std::vector<std::vector<int>> features = {
    //     {0, 1, 2, 4, 8},
    //     {0, 1, 2, 5, 9},
    //     {0, 1, 2, 6, 10},

    //     {1, 2, 3, 5, 9},
    //     {1, 2, 3, 6, 10},
    //     {1, 2, 3, 7, 11},

    //     {}

    // };

    const std::vector<std::vector<int>> features = {
        {0, 4, 8, 12, 13},
        {1, 5, 9, 13, 14},
        {1, 5, 9, 6, 10},
        {2, 6, 10, 7, 11},

        {12, 13, 14, 15, 11},
        {8, 9, 10, 11, 7},
        {8, 9, 10, 6, 5},
        {4, 5, 6, 2, 1},

        {15, 11, 7, 3, 2},
        {14, 10, 6, 2, 1},
        {14, 10, 6, 5, 9},
        {13, 9, 5, 8, 4},

        {3, 2, 1, 0, 4},
        {7, 6, 5, 4, 8},
        {7, 6, 5, 9, 10},
        {11, 10, 9, 13, 14},

        {3, 7, 11, 15, 14},
        {2, 6, 10, 14, 13},
        {2, 6, 10, 9, 5},
        {1, 5, 9, 8, 4},

        {0, 1, 2, 3, 7},
        {4, 5, 6, 7, 11},
        {4, 5, 6, 9, 10},
        {8, 9, 10, 13, 14},

        {12, 8, 4, 0, 1},
        {13, 9, 5, 1, 2},
        {13, 9, 5, 6, 10},
        {14, 10, 6, 7, 11},

        {15, 14, 13, 12, 8},
        {11, 10, 9, 8, 4},
        {11, 10, 9, 5, 6},
        {7, 6, 5, 1, 2}
    };
    // const std::vector<std::vector<int>> features = {
    //     {0, 1, 2, 3},
    //     {4, 5, 6, 7},
    //     {8, 9, 10, 11},
    //     {12, 13, 14, 15},
    //     {0, 4, 8, 12},
    //     {1, 5, 9, 13},
    //     {2, 6, 10, 14},
    //     {3, 7, 11, 15}
    // };
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
        for (long unsigned int i = 0; i < features.size(); i++) {
            value += net[i][extract_feature(after, features[i])];
        }
        return value;
    };

    typedef struct step {
        board state;
        board::reward reward;
        bool terminated;

    } Step;

    void td_0(Step last, Step next) {
        if (next.terminated){
            float current = estimate_value(next.state);
            float target = 0;
            float error = target - current;
            float adjust = alpha * error;
            for (long unsigned int i = 0; i < features.size(); i++) {
                net[i][extract_feature(next.state, features[i])] += adjust;
            }
        }

        float current = estimate_value(last.state);
        float target = estimate_value(next.state) + next.reward;
        float error = target - current;
        float adjust = alpha * error;
        for (long unsigned int i = 0; i < features.size(); i++) {
            net[i][extract_feature(last.state, features[i])] += adjust;
        }
    };
    void td_0_backward(Step last, float target){
        float current = estimate_value(last.state);
        float error = target - current;
        float adjust = alpha * error;
        for (long unsigned int i = 0; i < features.size(); i++) {
            net[i][extract_feature(last.state, features[i])] += adjust;
        }
    };

    virtual action take_action(const board& before) {
        int best_op = -1;
        int best_reward = -1;
        float best_value = -100000;
        board best_after = before;
        for (int op : {0, 1, 2, 3}) {
            board after = board(before);
            board::reward reward = after.slide(op);
            if (reward == -1) continue;
            float value = estimate_value(after);
            if (reward + value >= best_reward + best_value) {
                best_op = op;
                best_reward = reward;
                best_value = value;
                best_after = after;
            }
        }
        if(best_op != -1) history.push_back({best_after, best_reward, best_op == -1});
        // if (history.size() == 2) {
        //     td_0(history.front(), history.back());
        //     history.erase(history.begin());
        // }
        return action::slide(best_op);
    };
    virtual void open_episode(const std::string& flag = "") {
        history.clear();
    };
    virtual void close_episode(const std::string& flag = ""){
        if (history.empty() || alpha == 0) return;
        auto h = history.end()-1;
        td_0_backward(*h, 0);

        for (h--; h != history.begin() - 1; h--) {
            float target = (h + 1)->reward + estimate_value((h + 1)->state);
            td_0_backward(*h, target);
        }
    };

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
    std::vector<Step> history;
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
