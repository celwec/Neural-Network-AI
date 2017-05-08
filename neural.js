// Transposing matrices helper function
function transposeMatrix(matrix) {
  return matrix[0].map((col, i) => {
    return matrix.map((row) => {
      return row[i];
    });
  });
}

// Random number helper function
// Setting float to true will return a floating point number, otherwise integer
function randomRange(minimum = 0, maximum = 1, count = 1, float = false) {

  // Array of numbers returned at the end
  var numbers = [];

  // Until count has been reached, keep generating new numbers
  for (var i = 0; i < count; i++)
    numbers.push(roundFloat(Math.random() * (maximum - minimum), float == false ? 0 : 3) + minimum);

  return numbers;
}

// Rounding a floating number helper function
function roundFloat(float, i = 0) {

  // Multiplying float by 10 to the power of i
  float = float * Math.pow(10, i);

  // Rounding float to the nearest integer
  float = Math.round(float);

  // Dividing float by 10 to the power of i
  float = float / Math.pow(10, i);

  return float;
}

// Nonlinear helper function
function nonlin(i, derivative = false) {
  var o = [];

  i.forEach(v => {
    derivative ? o.push(v * (1 - v)) : o.push(1 / (1 + Math.exp(-v)));
  });

  return o;
}

// Matrix dot product helper function
function dot(matrix0, matrix1) {
  matrix1 = transposeMatrix(matrix1);

  var m0 = [];
  var c1 = 0;

  for (var r0 = 0; r0 < matrix0.length; r0++) {
    var m1 = [];
    for (var c0 = 0; c0 < matrix0[0].length; c0++) {
      m1.push(matrix0[r0][c0] * matrix1[0][c1]);
    }
    c1++;
    m0.push(m1);
  }

  return m0.map((e,i) => {
    return m0[i].reduce((a,b) => a+b);
  });
}

function multArrays(a, b) {
  return a.map((e, i) => {
    return e * b[i];
  });
}

function addArrays(a, b) {
  return a.map((e, i) => {
    return e + b[i];
  });
}

/*
** #############################################################################
** Actual Neural Network AI starts here
** #############################################################################
*/

// Training data for the ai
var training = [
  [0,0,1],
  [0,1,0],
  [0,1,1],
  [1,0,0]
];

// Expected outcome using the above training data
var expected = [
  [0,1,1,0]
];

// Transposing expected outcome to fit training data
expected = transposeMatrix(expected);

// Initializing weights randomly with mean 0
var syn0 = [];
for (var i = 0; i < training.length; i++)
  syn0.push(randomRange(-1, 1, expected[0].length, true));

console.log("Expected values:");
console.log(expected.map(e => e[0]));

var flag = false;

// Training a lot
for (var i = 0; i < 1000000; i++) {

  // Defining layers
  var l0 = training;
  var l1 = nonlin(dot(l0, syn0));

  if (!flag) {
    console.log("Neural Network AI guess before training:");
    console.log(syn0.map(e => e[0]));
    flag = !flag;
  }

  // By how much have we missed our expected values?
  var l1_error = expected.map((e, i) => {
    return e - l1[i];
  });

  // multiplying how much we missed by the slope of the sigmoid at the values l1
  var l1_delta = multArrays(l1_error, nonlin(l1, true));

  //
  l1_delta = l1_delta.map(e => {
    return [e];
  });

  var d = dot(l0, l1_delta);

  syn0 = syn0.map((e, i) => {
    return [e[0] + d[i]];
  });
}

console.log("Neural Network AI guess after training:");
console.log(l1);
