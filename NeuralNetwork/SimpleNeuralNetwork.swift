//
//  SimpleNeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Nicholas Cracchiolo on 1/22/17.
//  Copyright © 2017 Nicholas Cracchiolo. All rights reserved.
//

import Foundation

class SimpleNeuralNet {
	var weights:[Double] = []
	var alpha:Double = 0.01
	
	init(withSize size:Int) {
		for _ in 0..<size {
			weights.append(Double(arc4random_uniform(2)) - 1)
		}
	}
	func train(with inputs:[Double], answer:Int) {
		let output = feedForward(for: inputs)
		let error = Double(answer - output)
		for i in 0..<weights.count {
			weights[i] += alpha * error * inputs[i]
		}
	}
	func feedForward(for inputs:[Double]) -> Int {
		let sum = processPerceptron(for: inputs)
		return activatePerceptron(with: sum)
	}
	private func processPerceptron(for inputs:[Double]) -> Double {
		var sum:Double = 0
		for i in 0..<inputs.count {
			sum += inputs[i] * weights[i]
		}
		return sum
	}
	private func activatePerceptron(with value:Double) -> Int {
		if value > 0 { return 1 }
		else { return -1 }
	}
}
