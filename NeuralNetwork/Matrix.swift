//
//  Matrix.swift
//  NeuralNetwork
//
//  Created by Nicholas Cracchiolo on 1/22/17.
//  Copyright © 2017 Nicholas Cracchiolo. All rights reserved.
//

import Swift
import Accelerate

enum MatrixError: Error {
	case OutOfBounds
	case SizeMismatch(size1:(Int,Int),size2:(Int,Int))
}
protocol MatrixMath {
	func add(by:inout Double) -> Matrix
	func scale(by:inout Double) -> Matrix
	func divided(by denominator:inout Double) -> Matrix
	func divide(with numerator:inout Double) -> Matrix
	func exp() -> Matrix
	func multiply(m:Matrix) -> Matrix
	func transpose() -> Matrix
	func invert() -> Matrix
	func mean() -> Double
}
protocol MatrixVectorMath {
	func add(by matrix:Matrix) -> Matrix
	func subtract(by matrix:Matrix) -> Matrix
	func times(by matrix:Matrix) -> Matrix
	func divide(by matrix:Matrix) -> Matrix
}
struct Matrix {
	//Properties
	var rows:Int
	var cols:Int
	var size:(rows:Int,cols:Int)
	var data:[[Double]]
	var array:[Double]
	
	static func random(withRows rows:Int, cols:Int) -> Matrix {
		var m:Matrix = self.init(rows: rows, cols: cols)
		for _ in 0..<(rows * cols) {
			let rand = Double(arc4random_uniform(2)) - 1
			m.array.append(rand)
		}
		return m
	}
	
	//Various initalizers based on input types
	init(rows:Int,cols:Int) {
		self.rows = rows
		self.cols = cols
		self.size = (rows,cols)
		self.data = [[]]
		self.array = []
		
	}
	init(array:[Double],rows:Int,cols:Int) {
		self.rows = rows
		self.cols = cols
		self.size = (self.rows,self.cols)
		self.array = array
		self.data = [[]]
		var index = 0
		for r in 0..<rows {
			for c in 0..<cols {
				self.data[r][c] = array[index]
				index+=1
			}
		}
	}
	init(data:[[Double]],rows:Int,cols:Int) {
		self.rows = rows
		self.cols = cols
		self.size = (self.rows,self.cols)
		self.data = data
		self.array = []
		for r in 0..<rows {
			for c in 0..<cols {
				self.array.append(self.data[r][c])
			}
		}
	}
	
	//Get a specific column of the Matrix
	func get(column at:Int) -> Matrix {
		let m = Matrix(array:self.data[at], rows:self.rows, cols:1)
		return m
	}
}

extension Matrix : MatrixVectorMath {
	func add(by matrix:Matrix) -> Matrix {
		var addition = [Double](repeating : 0.0, count : self.array.count)
		vDSP_vaddD(self.array, 1, matrix.array, 1, &addition, 1, vDSP_Length(self.array.count))
		let m = Matrix(array: addition, rows: self.rows, cols: self.cols)
		return m
	}
	func subtract(by matrix:Matrix) -> Matrix {
		var subtraction = [Double](repeating: 0.0, count: self.array.count)
		vDSP_vsubD(self.array, 1, matrix.array, 1, &subtraction, 1, vDSP_Length(self.array.count))
		let newMatrix = Matrix(array: subtraction, rows: self.rows, cols: self.cols)
		return newMatrix
	}
	func times(by matrix:Matrix) -> Matrix {
		var scaled = [Double](repeating : 0.0, count : self.array.count)
		vDSP_vmulD(self.array, 1, matrix.array,1, &scaled, 1, vDSP_Length(self.array.count))
		let m = Matrix(array: scaled, rows: self.rows, cols: self.cols)
		return m
	}
	func divide(by matrix:Matrix) -> Matrix {
		var divided = [Double](repeating : 0.0, count : self.array.count)
		vDSP_vdivD(self.array, 1, matrix.array, 1, &divided, 1, vDSP_Length(self.array.count))
		let m = Matrix(array: divided, rows: self.rows, cols: self.cols)
		return m
	}
}


extension Matrix : MatrixMath {
	func add(by:inout Double) -> Matrix {
		var addition = [Double](repeating : 0.0, count : self.array.count)
		vDSP_vsaddD(self.array, 1, &by, &addition, 1, vDSP_Length(self.array.count))
		let newMatrix = Matrix(array: addition, rows: self.rows, cols: self.cols)
		return newMatrix
	}
	func subtract(by:inout Double) -> Matrix {
		var subtraction = [Double](repeating: 0.0, count: self.array.count)
		vDSP_vsubD(self.array, 1, &by, 1, &subtraction, 1, vDSP_Length(self.array.count))
		let newMatrix = Matrix(array: subtraction, rows:self.rows, cols: self.cols)
		return newMatrix
	}
	func scale(by:inout Double) -> Matrix {
		var scaled = [Double](repeating : 0.0, count : self.array.count)
		vDSP_vsmulD(self.array, 1, &by, &scaled, 1, vDSP_Length(self.array.count))
		let newMatrix = Matrix(array: scaled, rows: self.rows, cols: self.cols)
		return newMatrix
	}
	func divided(by denominator:inout Double) -> Matrix {
		var divided = [Double](repeating : 0.0, count : self.array.count)
		vDSP_vsdivD(self.array, 1, &denominator, &divided, 1, vDSP_Length(self.array.count))
		let newMatrix = Matrix(array: divided, rows: self.rows, cols: self.cols)
		return newMatrix
	}
	func divide(with numerator:inout Double) -> Matrix {
		var divided = [Double](repeating : 0.0, count : self.array.count)
		vDSP_svdivD(self.array, &numerator, 1, &divided, 1, vDSP_Length(self.array.count))
		let newMatrix = Matrix(array: divided, rows: self.rows, cols: self.cols)
		return newMatrix
	}
	func times(with matrix:Matrix) -> Matrix {
		var mult = [Double](repeating : 0.0, count : self.array.count)
		vDSP_vmulD(self.array, 1, matrix.array, 1, &mult, 1, vDSP_Length(self.array.count))
		return Matrix(array: mult, rows: self.rows, cols: self.cols)
	}
	func exp() -> Matrix {
		var n = Int32(self.array.count)
		var y = self.array
		vvexp(&y, self.array, &n)
		return Matrix(array: y, rows: self.rows, cols: self.cols)
	}
	func multiply(m: Matrix) -> Matrix {
		var matrixMult = [Double](repeating : 0.0, count : self.array.count)
		vDSP_mmulD(self.array, 1, m.array, 1, &matrixMult, 1, vDSP_Length(self.rows), vDSP_Length(m.cols), vDSP_Length(self.cols))
		return Matrix(array: matrixMult, rows: self.rows, cols: m.cols)
	}
	func transpose() -> Matrix {
		var transMatrix = [Double](repeating : 0.0, count : self.array.count)
		vDSP_mtransD(self.array, 1, &transMatrix, 1, vDSP_Length(self.cols), vDSP_Length(self.rows))
		return Matrix(array: transMatrix, rows: self.cols, cols: self.rows)
	}
	func invert() -> Matrix {
		var inverted = self.array
		var N = __CLPK_integer(sqrt(Double(inverted.count)))
		var pivots = [__CLPK_integer](repeating: 0, count: Int(N))
		var workspace = [Double](repeating: 0.0, count: Int(N))
		var error : __CLPK_integer = 0
		dgetrf_(&N, &N, &inverted, &N, &pivots, &error)
		dgetri_(&N, &inverted, &N, &pivots, &workspace, &N, &error)
		return Matrix(array: inverted, rows: self.rows, cols: self.cols)
	}
	func mean() -> Double {
		var mean:Double = 0.0
		vDSP_meanvD(self.array,1,&mean,vDSP_Length(self.array.count))
		return mean
	}
}
