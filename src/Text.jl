# text.jl
#
# author: Wade Shen
# swade@ll.mit.edu
# Copyright &copy; 2014 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
module Text
using DataStructures, Devectorize, Ollam, Stage, GZip
import Base: start, done, next

# -------------------------------------------------------------------------------------------------------------------------
# module-wide utilities
# -------------------------------------------------------------------------------------------------------------------------
getindex{K}(set :: Set{K}, key :: K) = contains(set, key) # sets don't have getindex method

include("constants.jl")
include("readers.jl")
include("tokenizers.jl")
include("features.jl")
include("models.jl")
include("lid.jl")

end
