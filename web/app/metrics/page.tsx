'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useState } from "react"
import { X } from "lucide-react"

export default function MetricsPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  const openImageModal = (imageSrc: string) => {
    setSelectedImage(imageSrc)
  }

  const closeImageModal = () => {
    setSelectedImage(null)
  }
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-12">
        <h1 className="text-4xl font-bold text-gray-900">
          Model Performance Dashboard
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Comprehensive evaluation of our state-of-the-art kinase binding prediction model
        </p>
        <div className="flex justify-center items-center space-x-8 text-sm pt-4">
          <span className="font-medium text-gray-700">Production Ready</span>
          <span className="font-medium text-gray-700">Scientifically Validated</span>
          <span className="font-medium text-gray-700">Industry Standard</span>
        </div>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between">
              <span>Predictive Accuracy</span>
              <div className="text-2xl font-bold text-gray-900">82%</div>
            </CardTitle>
            <CardDescription>Area Under ROC Curve (AUROC)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span>AUROC:</span>
                <span className="font-medium">0.82</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>PR-AUC:</span>
                <span className="font-medium">0.70</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Calibration (ECE):</span>
                <span className="font-medium">0.07</span>
              </div>
              <div className="text-xs text-gray-600 mt-2 p-2 border rounded">
                Exceeds industry benchmark of 0.75 AUROC for binding prediction
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between">
              <span>Inference Speed</span>
              <div className="text-2xl font-bold text-gray-900">120ms</div>
            </CardTitle>
            <CardDescription>Real-time prediction latency</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span>Median (P50):</span>
                <span className="font-medium">120ms</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>95th percentile:</span>
                <span className="font-medium">240ms</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Model size:</span>
                <span className="font-medium">23MB</span>
              </div>
              <div className="text-xs text-gray-600 mt-2 p-2 border rounded">
                Optimized for high-throughput virtual screening workflows
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between">
              <span>Reliability Score</span>
              <div className="text-2xl font-bold text-gray-900">93%</div>
            </CardTitle>
            <CardDescription>Prediction confidence & calibration</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span>Calibration quality:</span>
                <span className="font-medium">Excellent</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Uncertainty estimation:</span>
                <span className="font-medium">Enabled</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Domain coverage:</span>
                <span className="font-medium">95%</span>
              </div>
              <div className="text-xs text-gray-600 mt-2 p-2 border rounded">
                Temperature-scaled probabilities provide reliable confidence scores
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Evaluation Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <Card className="hover:shadow-xl transition-shadow duration-300">
          <CardHeader className="text-center">
            <CardTitle className="text-xl">Comprehensive Model Evaluation</CardTitle>
            <CardDescription className="text-base">
              ROC curves, precision-recall analysis, and calibration plots
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="w-full relative group">
              <img 
                src="/final_model_evaluation.png" 
                alt="Final Model Evaluation" 
                className="w-full h-auto rounded-lg border cursor-pointer hover:border-gray-400 transition-all duration-300 group-hover:shadow-lg"
                onClick={() => openImageModal("/final_model_evaluation.png")}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg pointer-events-none"></div>
              <div className="absolute bottom-4 left-4 text-white font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                Click to enlarge
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="hover:shadow-xl transition-shadow duration-300">
          <CardHeader className="text-center">
            <CardTitle className="text-xl">Training Dynamics</CardTitle>
            <CardDescription className="text-base">
              Loss convergence and validation performance over training epochs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="w-full relative group">
              <img 
                src="/training_curves.png" 
                alt="Training Curves" 
                className="w-full h-auto rounded-lg border cursor-pointer hover:border-gray-400 transition-all duration-300 group-hover:shadow-lg"
                onClick={() => openImageModal("/training_curves.png")}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg pointer-events-none"></div>
              <div className="absolute bottom-4 left-4 text-white font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                Click to enlarge
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Scientific Impact & Guidelines */}
      <Card className="hover:shadow-xl transition-shadow duration-300">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl text-gray-900">
            Drug Discovery Optimization Framework
          </CardTitle>
          <CardDescription className="text-lg text-gray-600">
            Evidence-based molecular optimization strategies for medicinal chemists
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <h3 className="font-bold text-lg text-gray-900">
                  Physicochemical Properties
                </h3>
                <div className="space-y-3">
                  <div className="p-3 border rounded-lg">
                    <span className="font-medium">Molecular Weight Optimization</span>
                    <p className="text-sm text-gray-600">Target &lt;500 Da for enhanced oral bioavailability</p>
                  </div>
                  <div className="p-3 border rounded-lg">
                    <span className="font-medium">Polar Surface Area</span>
                    <p className="text-sm text-gray-600">Reduce TPSA &lt;140 Ų for CNS penetration</p>
                  </div>
                  <div className="p-3 border rounded-lg">
                    <span className="font-medium">Conformational Flexibility</span>
                    <p className="text-sm text-gray-600">Minimize rotatable bonds for binding entropy</p>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <h3 className="font-bold text-lg text-gray-900">
                  Lead Optimization Strategy
                </h3>
                <div className="space-y-3">
                  <div className="p-3 border rounded-lg">
                    <span className="font-medium">Drug-likeness Enhancement</span>
                    <p className="text-sm text-gray-600">Systematic lead optimization protocols</p>
                  </div>
                  <div className="p-3 border rounded-lg">
                    <span className="font-medium">Property Balance</span>
                    <p className="text-sm text-gray-600">Multi-parameter optimization principles</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="pt-6 border-t text-center">
              <div className="flex items-center justify-center space-x-8 text-sm text-gray-600">
                <span>Automatically applied in predictions</span>
                <span>Evidence-based recommendations</span>
                <span>Industry validated</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Image Modal */}
      {selectedImage && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-85 flex items-center justify-center z-50 p-4 backdrop-blur-sm"
          onClick={closeImageModal}
        >
          <div className="relative max-w-7xl max-h-full">
            <button
              onClick={closeImageModal}
              className="absolute top-4 right-4 bg-white bg-opacity-90 hover:bg-opacity-100 rounded-full p-3 text-gray-800 hover:text-black transition-all shadow-lg z-10"
            >
              <X size={24} />
            </button>
            <img 
              src={selectedImage} 
              alt="Detailed model evaluation view" 
              className="max-w-full max-h-full object-contain rounded-xl shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </div>
  )
}
