# readers.jl
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
export read_tweets, read_usenet, filelines, zopen

# -------------------------------------------------------------------------------------------------------------------------
# Basic utilities
# -------------------------------------------------------------------------------------------------------------------------
function zopen(fn :: AbstractString)
  return ismatch(r"^.*\.gz$", fn) ? gzopen(fn) : open(fn)
end

type FileLines
  name :: AbstractString
end

start(itr :: FileLines) = zopen(itr.name)
function done(itr :: FileLines, stream)
  if !eof(stream)
    return false
  end
  close(stream)
  return true
end
function next(itr :: FileLines, stream) 
  x = readline(stream)
  return (x, stream)
end
eltype(itr :: FileLines) = ByteString


# get a file line iterator from a file name, open with gzip as needed
filelines(fn :: AbstractString) = FileLines(fn)
streamlines(f) = eachline(f) # convenience

#function getfile(name)
#    file = joinpath(savedir, name)
#    if !isfile(file)
#        file = download(urlbase*name, file)
#    end
#    file
#end

# -------------------------------------------------------------------------------------------------------------------------
# Text format readers
# -------------------------------------------------------------------------------------------------------------------------

# read collection of tweets from a file
function read_tweets(fn :: AbstractString; stopList=english_stoplist, header=news_email_header, tokenizer=tenglish_tokenizer, limit=-1, keepFn=x->true, lg=STDERR)
  ret = Dict{String, Float32}[]
  rlat = Float32[]
  rlong = Float32[]
  
  tic()
  f = zopen(fn)
  tweets = JSON.parse(f)
  close(f)

  for tweet in tweets #filelines(fn)
    # json parser is wicked slow
    #tweet = parse(l)
    text = tweet["text"]
    if keepFn(text) # was vec (see below)
      
      # extract text and location
      # m = match(r"^.*?\"?text\"?\s*:\s*\"(.*?)\".*$", l)
      # if (m == nothing) continue; end
      # text = m.captures[1]
      # text = unescape_string(text)
      valid = true
      
      # validate text
      for c in text
        if 0xd800 <= c <= 0xdfff || 0x10ffff < c # same check made by isvalid(Char,ch) and deprecated is_valid_char
          valid = false
        end 
      end
      
      # check for valid geo
      #m = match(r"^.*?\"?geo\"?\s*:\s*(\{.*?\}|null).*$", l)
      #geo = m.captures[1]
      #lat, long = -1000, -1000
      geo = tweet["geo"]
      lat, long = -1000, -1000
      if (geo != nothing && !haskey(geo, "coordinates"))
        #mm = match(r"^{.*\"?coordinates\"?\s*:\s*\[(.*?)\s*,\s*(.*?)\s*\]\s*.*\}$", geo)
        lat  = geo["coordinates"][1]
        long = geo["coordinates"][2]
        #if (m != nothing)
        #  lat, long = mm.captures
        #end
      end
      
      if (valid)
        text = replace(text, r"\s+"s, " ")
        vec = Dict{String, Float32}()
        for w in filter(x -> !stopList[x], tokenizer(lowercase(strip(text))))
          if haskey(vec, w)
            vec[w] += 1
          else
            vec[w] = 1
          end
        end
        
        push!(ret, vec)
        push!(rlat, float32(lat))
        push!(rlong, float32(long))
      end
      if (limit > 0 && size(ret, 1) > limit)
        break
      end
    end
  end
  t = toq()
  @info lg @sprintf("finished reading %60s [n = %6d, time = %10.3f]", fn, length(ret), t)
  return ret, rlat, rlong
end

# usenet/email single document reader -- 20ng
function read_usenet(fn :: AbstractString; stopList=english_stoplist, header=news_email_header, tokenizer=english_tokenizer, lg=STDERR)
  ignore = false
  @info lg @sprintf("reading: %s", fn)
  vec = Dict{String, Float32}()
  for l in eachline(`iconv -f latin1 -t utf8 $fn`)
    cl = lowercase(strip(l))
    if (ismatch(r"--+", cl)) ignore = true end
    if (!ignore && cl != "" && !ismatch(r"^((jj)?>|:|\$).*$", cl) && !ismatch(header, cl))
      for w in tokenizer(cl) #split(cl, r"\s+")
        if !stopList[w]
          if (has(vec, w)) 
            vec[w] += 1
          else 
            vec[w] = 1 
          end
        end
      end
    end
  end
  return [ vec ]
end
