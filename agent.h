/**
 * Framework for 2048 & 2048-like Games (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
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

   protected:
    virtual void init_weights(const std::string& info) {
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
class player : public random_agent {
   public:
    player(const std::string& args = "")
        : random_agent("name=dummy role=player " + args),
          opcode({0, 1, 2, 3}),
          mode(args) {}

    virtual action take_action(const board& before) {
        std::shuffle(opcode.begin(), opcode.end(), engine);
        if (mode == "") {
            for (int op : opcode) {
                board::reward reward = board(before).slide(op);
                if (reward != -1) return action::slide(op);
            }
            return action();
        } else if (mode == "greedy") {
            int max = 0;
            int op_to_send = 0;
            for (int op : opcode) {
                board::reward reward = board(before).slide(op);
                if (reward != -1 && reward >= max) {
                    max = reward;
                    op_to_send = op;
                }
            }
            return action::slide(op_to_send);
        } else if (mode == "heuristic") {
            float max = 0;
            int op_to_send = 0;
            for (int op : opcode) {
                float score = 0;
                board after = board(before);
                board::reward reward = after.slide(op);
                if (reward != -1) {
                    board::grid& tile = after;
                    unsigned int max_elem = 0;
                    int max_at_corner = 0, space_num = 0, monotonic_decreasing = 0, decreasing = 0;

                    // check whether max element is at (0,0).
                    for (int i = 0; i < 16; i++)
                        max_elem = (after(i) > max_elem) ? after(i) : max_elem;
                    max_at_corner = (after(0) == max_elem);

                    // check # of space and row decreasing.
                    for (auto& row : tile) {
                        bool flag = 1, dflag = 1;
                        for (int c = 0; c < 4; c++) {
                            space_num += (row[c] == 0) ? 1 : 0;
                            if (c < 3) {
                                flag *= (row[c] >= row[c + 1]);
                                dflag *= (row[c] > row[c + 1]);
                            }
                        }
                        monotonic_decreasing += flag;
                        decreasing += dflag;
                    }
                    // check column decreasing.
                    for (int j = 0; j < 3; j++) {
                        bool flag = 1, dflag = 1;
                        for (int i = 0; i < 4; i++) {
                            flag *= (tile[i][j] >= tile[i][j + 1]);
                            dflag *= (tile[i][j] > tile[i][j + 1]);
                        }
                        monotonic_decreasing += flag;
                        decreasing += dflag;
                    }

                    score = 1.0 * reward + 60.0 * max_at_corner + 0 * space_num + .27 * monotonic_decreasing - .05 * decreasing;
                    if (score >= max) {
                        max = score;
                        op_to_send = op;
                    }
                }
            }
            return action::slide(op_to_send);
        }
        return action();
    }

   private:
    std::array<int, 4> opcode;
    std::string mode;
};
