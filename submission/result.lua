----------------------------------------------------------------------
-- A torch script for testing the model for homework 1
-- For DS-GA-1008, Deep Learning
-- New York University
--
-- Written by Emil Christensen
-- based on 5_test.lua written by Clement Farabet
-- 02/08/2015
----------------------------------------------------------------------

-- Import libraries
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'      -- provides a normalization operator
require 'image'   -- for color transforms

-- Use 4 threads
torch.setnumthreads(4)

----------------------------------------------------------------------
-- Download files as necessary --

-- Define URLs
model_url = 'http://cims.nyu.edu/~erc399/'
model_file = 'deep_blue_model.net'
mean_file = "mean.lua"
std_file = "std.lua"

data_url = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'
test_file = 'test_32x32.t7'
train_file = 'train_32x32.t7'


-- Download files unless already present
if not paths.filep(model_file) then
   print('==> downloading model.net file')
   os.execute('wget ' .. model_url .. model_file)
end
if not paths.filep(train_file) then
   print('==> downloading train data file')
   os.execute('wget ' .. data_url .. train_file)
end
if not paths.filep(test_file) then
   print('==> downloading test data file')
   os.execute('wget ' .. data_url .. test_file)
end


if not paths.filep(mean_file) then
   print('==> downloading training mean file')
   os.execute('wget ' .. model_url .. mean_file)
end
if not paths.filep(std_file) then
   print('==> downloading training std file')
   os.execute('wget ' .. model_url .. std_file)
end
----------------------------------------------------------------------
-- Preprocessing -- 
----------------------------------------------------------------------

tesize = 26032

-- Perform preprocessing as in 1_data.lua
print '==> loading dataset'

loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return tesize end
}

print '==> preprocessing data'

testData.data = testData.data:float()

print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

channels = {'y','u','v'}

print '==> preprocessing data: normalize each feature (channel) globally'

-- Load mean and std values saved from 1_data.lua 
-- This saves unecessary computation

mean = torch.load(mean_file)
std = torch.load(std_file)

for i,channel in ipairs(channels) do
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

print '==> preprocessing data: normalize all three channels locally'

neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

for c in ipairs(channels) do
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

----------------------------------------------------------------------
-- Evaluate model and save predictions -- 

-- Load the model
print('==> loading model')
model = torch.load(model_file)
model:evaluate()

-- Open file for saving predictions
file = io.open("predictions.csv","w")
file:write("Id,Prediction")

print('==> testing')
-- test over test data
for t = 1,testData:size() do

   -- display progress
   xlua.progress(t, testData:size())

   -- get new sample
   local input = testData.data[t]
   input = input:double()
   local target = testData.labels[t]

   -- test sample
   local pred = model:forward(input)

   -- pick out model's  choice
   m = pred:max()
   idx = torch.linspace(1,pred:size(1),pred:size(1))
   p = idx[pred:eq(m)]

   -- add to predictions
   file:write("\n",t,",",p[1])
end

-- Write to file and close
print '==> writing results to file'
file:flush()
file:close()
