// create main global object
var trimapjs = trimapjs || { REVISION: 'ALPHA' };

(function(global) {
  "use strict";

    var assert = function(condition, message) {
    if (!condition) { throw message || "Assertion failed"; }
  }

  var getopt = function(opt, field, defaultval) {
    if(opt.hasOwnProperty(field)) {
      return opt[field];
    } else {
      return defaultval;
    }
  }



// return 0 mean unit standard deviation random number
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) {
      return_v = false;
      return v_val;
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this for next function call for efficiency
    return_v = true;
    return u*c;
  }

  // return random normal number
  var randn = function(mu, std){ return mu+gaussRandom()*std; }


  Array.prototype.reshape = function(rows, cols) {
  var copy = this.slice(0); // Copy all elements.
  this.length = 0; // Clear out existing array.

  for (var r = 0; r < rows; r++) {
    var row = [];
    for (var c = 0; c < cols; c++) {
      var i = r * cols + c;
      if (i < copy.length) {
        row.push(copy[i]);
      }
    }
    this.push(row);
  }
};



  var bincount = function(X,Y){
      var n = Math.max(...X)
      var t = X.length
      var Z = Array(n+1).fill(0)
      for(var i = 0; i < t; i++){
          var k = X[i]
          Z[k] = Z[k] + Y[i]
      }
      return Z
  }

  var trimap_grad = function(Y, triplets, weights){
      var lY = Y
      var nlength = lY.length
      var dim = lY[0].length
      var m = triplets.length
        //grad are nj array
      var grad = Array(nlength*dim).fill(0);


        // triplets is a m*3 js array, y_i,j,k are m*dim js array, Y is a n*dim  nj array
      var y_i = []
      var y_j = []
      var y_k = []

      for (var i = 0; i < m; ++i) {
          var x1 = triplets[i][0]
          var x2 = lY[x1]
          y_i.push(x2)

          var y1 = triplets[i][1]
          var y2 = lY[y1]
          y_j.push(y2)

          var z1 = triplets[i][2]
          var z2 = lY[z1]
          y_k.push(z2)
      }
        //y_i,j,k are now m*dim nj array


      y_i = nj.array(y_i)
      y_j = nj.array(y_j)
      y_k = nj.array(y_k)


      var y_ij = y_i.subtract(y_j)
      var y_ik = y_i.subtract(y_k)

      var my_ij = y_ij.multiply(y_ij).tolist()



      var y_ijx = m
      var y_ijy = dim
      var d_ij = []

      for(var i = 0; i < y_ijx; ++i){
          var k = 0
          for(var j = 0; j<y_ijy; ++j){
              k = k + my_ij[i][j]
          }
          k = k+1
          d_ij.push(k)
      }

//dij dik are both m*1 js array
      var my_ik = y_ik.multiply(y_ik).tolist()
      var y_ijx = m
      var y_ijy = dim
      var d_ij = []

      for(var i = 0; i < y_ijx; ++i){
          var k = 0
          for(var j = 0; j<y_ijy; ++j){
              k = k + my_ij[i][j]
          }
          k = k+1
          d_ij.push(k)
      }
        //dij dik are both m*1 js array
      var y_ikx = my_ik.length
      var y_iky = my_ik[0].length
      var d_ik = []

      for(var i = 0; i < y_ikx; ++i){
          var k = 0
          for(var j = 0; j<y_iky; ++j){
              k = k + my_ik[i][j]
          }
          k = k+1
          d_ik.push(k)
      }

      var num_viol = 0
      var denom = []
        //denom is a m*1 js array
      for(var i = 0; i< y_ikx; i++){
          var k = 0

          if(d_ij[i]>d_ik[i]){
              num_viol +=1
          }

          k = d_ij[i] + d_ik[i]

          k = k*k
          denom.push(k)
      }

      var loss = 0
      for(var i = 0; i < weights.length; i++){
          var k = d_ij[i]/(d_ij[i]+d_ik[i])
          k = k * weights[i]
          loss = loss + k
      }
        //gs
            y_ij = y_ij.tolist()
      var ylength = y_ij.length
      for(var i = 0; i < ylength; i++){
          var n = 2 * (d_ik[i] / denom[i] * weights[i])
          for(var j = 0; j < dim; j++){
              y_ij[i][j] = y_ij[i][j] * n
          }
      }
        //go
            y_ik = y_ik.tolist()
      for(var i = 0; i < ylength; i++){

          var n = 2 * (d_ij[i] / denom[i] * weights[i])
          for(var j = 0; j < dim; j++){
              y_ik[i][j] = y_ik[i][j] * n
          }
      }

      for(var i =0;i < dim; i++){
          var X1 = []
          var Y1 = []
          for(var j = 0;j < m; j++){
              X1.push(triplets[j][0])
              var Y11 = y_ij[j][i]
              var Y12 = y_ik[j][i]
              Y1.push(Y11-Y12)
          }
          var bin1 = bincount(X1,Y1)



          var X2 = []
          var Y2 = []
          for(var j = 0;j < m; j++){
              X2.push(triplets[j][1])
              var Y21 = y_ij[j][i]
              Y2.push(0 - Y21)
          }
          var bin2 = bincount(X2,Y2)



          var X3 = []
          var Y3 = []
          for(var j = 0;j < m; j++){
              X3.push(triplets[j][2])
              var Y32 = y_ik[j][i]
              Y3.push(Y32)
          }
          var bin3 = bincount(X3,Y3)



          for(var j = 0; j < nlength; j++){
              grad[j*dim+i] += bin1[j]
              grad[j*dim+i] += bin2[j]
              grad[j*dim+i] += bin3[j]
          }

      }

      grad.reshape(nlength,dim);

    return{"cost": loss,
          "gradiant": grad,
          "numviol": num_viol
    }
  }





  var triMap = function(opt) {
    var opt = opt || {};
    this.perplexity = getopt(opt, "perplexity", 30); // effective number of nearest neighbors
    this.nd = getopt(opt, "nd", 2); // by default 2-D tSNE
    this.epsilon = getopt(opt, "epsilon", 10); // learning rate

    this.eta=2000.0
    this.iter = 0;
  }

  triMap.prototype = {

    // this function takes a set of high-dimensional points
    // and creates matrix P from them using gaussian kernel
    initDataRaw: function(X,a,b) {
        var n = X.length;
        var dim = X[0].length;


        this.n = n
        this.dim = dim

        this.Y = []
        for(var Yn=0;Yn < this.n;Yn++){
            for(var Ynd=0;Ynd < this.nd;Ynd++){
                this.Y.push(randn(0.0, 1e-4))
            }
        }
        this.Y.reshape(this.n,this.nd)
        this.cost = Infinity
        this.best_cost = Infinity
        this.best_Y = this.Y
        this.tol = 1e-7
        this.num_iter = 1000
        // triplets are js array
        this.triplets = a
        // weights are js array
        this.weights = b
        this.num_triplets = a.length

    },


    // return pointer to current solution
    getSolution: function() {
      return {"Y": this.Y,
          "loss": this.cost

    };
    },


    // perform a single step of optimization to improve the embedding
    step: function() {

        var old_cost = this.cost
        var tri = trimap_grad(this.Y, this.triplets, this.weights)
        this.cost = tri.cost
        this.grad = tri.gradiant
        this.num_viol = tri.numviol


        if(this.cost < this.best_cost) {
            this.best_cost = this.cost
            this.best_Y = this.Y
        }

        var njgrad = this.grad
        for(var Yn=0;Yn < this.n;Yn++){
            for(var Ynd=0;Ynd < this.nd;Ynd++){
                njgrad[Yn][Ynd] = njgrad[Yn][Ynd] * this.eta / this.num_triplets * this.n
                this.Y[Yn][Ynd] = this.Y[Yn][Ynd] - njgrad[Yn][Ynd]
            }
        }


        if(old_cost > this.cost + this.tol){
            this.eta = this.eta * 1.01
        }
        else{
            this.eta = this.eta * 0.5
        }


    },

  }

  global.triMAP = triMap; // export tSNE class
})(trimapjs);


// export the library to window, or to module in nodejs
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.trimapjs = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(trimapjs);