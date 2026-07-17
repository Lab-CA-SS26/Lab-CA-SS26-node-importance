/** @file bavarian.cpp
 * Driver for the Bavarian approximation framework for betweenness centrality
 * with static sampling.
 *
 * @author Matteo Riondato
 * @date 2020 04 20
 *
 * @copyright
 *
 * Copyright 2020 Matteo Riondato <rionda@acm.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bavarian.hpp"

int main(int argc, char **argv) {
  return doMain<Bavarian::Sampling::statix>(argc, argv);
}
