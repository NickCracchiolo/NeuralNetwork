//
//  MultilayeredNeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Nicholas Cracchiolo on 1/22/17.
//  Copyright © 2017 Nicholas Cracchiolo. All rights reserved.
//

import Foundation

class MultilayeredNeuralNet {
	var weights:[Matrix] = []
	var alpha:Double = 0.01
	var layers:[Int] = []
	var outputs:[Matrix] = []
	
	/*
	 * MARK: Init with layers with each element in array corresponding with
	 * the number of neurons for that layer
	 */
	init(with layers:[Int]) {
		self.layers = layers
		for _ in 0..<layers.count {
			weights.append(Matrix.random(withRows: 10, cols: 10))
			//weights.append(Double(arc4random_uniform(2)) - 1)
		}
	}
	
	func train(with input:Matrix, answer:Matrix) {
		let length = outputs.count
		let output = feedForward(for: input)
		var error = outputError(actual: answer, output: output)
		for i in 0..<length {
			let index = length - i
			var d = delta(error: error, output: outputs[length - i])
			let alphaDelta = d.scale(by: &alpha)
			weights[index] = weights[i].add(by: alphaDelta)
			error = internalError(delta: alphaDelta, weights: weights[index])
		}
	}
	
	/*
	 * MARK: Feedforward
	 *
	 * Transposes the inital input so to then multiply by the corresponding 
	 * weights for that layer. It then takes the output and runs it throught 
     * the sigmoid function. It then takes that output and multiplies it by
	 * the next layer's weights. Returns the output Matrix
	 */
	func feedForward(for input:Matrix) -> Matrix {
		var out = input.transpose()
		for i in 0..<layers.count {
			let transposedInput = out
			let product = transposedInput.multiply(m: weights[i])
			out = sigmoid(sum: product)
			outputs.append(out)
		}
		return out
	}
	
	/*
	 * MARK: Sigmoid computation
	 *
	 * Applys the sigmoid function: 1/(1 + exp(-(sum*weights))) to each value
	 * in the inputted matrix
	 */
	private func sigmoid(sum:Matrix) -> Matrix {
		let exp:Matrix = sum.exp()
		var one:Double = 1.0
		let addOne = exp.add(by:&one)
		return addOne.divide(with: &one)
	}
	
	/*
	* MARK: Sigmoid Prime computation
	*
	* Applys the sigmoid derivative function: sum*(1-sum) to each value in the
	* matrix
	*/
	private func sigmoidPrime(sum:Matrix) -> Matrix {
		let sig = sigmoid(sum: sum)
		var one:Double = 1.0
		var negOne:Double = -1.0
		var negSig = sig.scale(by: &negOne)
		negSig = negSig.add(by: &one)
		return sig.times(with: negSig)
	}
	
	private func delta(error:Matrix, output:Matrix) -> Matrix {
		let sig = sigmoidPrime(sum: output)
		return error.times(with: sig)
	}
	private func internalError(delta:Matrix, weights:Matrix) -> Matrix {
		let trans = weights.transpose()
		return delta.multiply(m: trans)
	}
	private func outputError(actual:Matrix, output:Matrix) -> Matrix {
		let diff = actual.subtract(by: output)
		return diff
	}
}
