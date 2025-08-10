'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function DataPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Dataset Information</h1>
        <p className="text-muted-foreground mt-2">
          Details about the training data and experimental setup
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Dataset Card</CardTitle>
          <CardDescription>BindingDB Kinase Top-10 Subset</CardDescription>
        </CardHeader>
        <CardContent className="prose prose-sm max-w-none">
          <h3>Label Policy</h3>
          <p>
            Binary labels based on binding affinity: pX ≥ 7 (positive binding), pX ≤ 5 (negative binding).
            Data points between 5-7 were excluded to reduce label noise.
          </p>

          <h3>Deduplication</h3>
          <p>
            Duplicate target+SMILES pairs were resolved by keeping the entry with strongest binding affinity.
          </p>

          <h3>Data Split</h3>
          <p>
            Scaffold-based splitting using hash function to ensure molecular diversity between train/val/test sets.
            See scaffold splitting script in repository for details.
          </p>

          <h3>Ethics Note</h3>
          <p className="text-muted-foreground">
            This tool is intended for screening and research purposes only. 
            Results should not be used for medical decisions or direct patient care.
          </p>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Training Set</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Total samples:</span>
                <span className="font-medium">3,892</span>
              </div>
              <div className="flex justify-between">
                <span>Positive rate:</span>
                <span className="font-medium">58.2%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Validation Set</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Total samples:</span>
                <span className="font-medium">413</span>
              </div>
              <div className="flex justify-between">
                <span>Positive rate:</span>
                <span className="font-medium">57.8%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Test Set</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Total samples:</span>
                <span className="font-medium">471</span>
              </div>
              <div className="flex justify-between">
                <span>Positive rate:</span>
                <span className="font-medium">58.6%</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Dataset Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Molecular Properties Distribution</CardTitle>
            <CardDescription>Key physicochemical properties across the dataset</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center">
                <span>Molecular Weight</span>
                <span className="text-muted-foreground">150-800 Da (avg: 420 Da)</span>
              </div>
              <div className="flex justify-between items-center">
                <span>LogP</span>
                <span className="text-muted-foreground">-2.5 to 8.2 (avg: 3.1)</span>
              </div>
              <div className="flex justify-between items-center">
                <span>TPSA</span>
                <span className="text-muted-foreground">0-250 Ų (avg: 85 Ų)</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Heavy Atoms</span>
                <span className="text-muted-foreground">8-58 (avg: 28)</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Rotatable Bonds</span>
                <span className="text-muted-foreground">0-18 (avg: 6)</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Target Kinase Distribution</CardTitle>
            <CardDescription>Top kinase targets in the dataset</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center">
                <span>CDK2</span>
                <span className="text-muted-foreground">892 compounds</span>
              </div>
              <div className="flex justify-between items-center">
                <span>EGFR</span>
                <span className="text-muted-foreground">756 compounds</span>
              </div>
              <div className="flex justify-between items-center">
                <span>PKA</span>
                <span className="text-muted-foreground">624 compounds</span>
              </div>
              <div className="flex justify-between items-center">
                <span>SRC</span>
                <span className="text-muted-foreground">581 compounds</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Others (6 targets)</span>
                <span className="text-muted-foreground">1,923 compounds</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Quality and Validation */}
      <Card>
        <CardHeader>
          <CardTitle>Data Quality & Validation</CardTitle>
          <CardDescription>Quality control measures and validation protocols</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-3 text-blue-700">Quality Control</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                  <span>SMILES standardization using RDKit</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                  <span>Removal of invalid molecular structures</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                  <span>Outlier detection based on binding affinity</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2" />
                  <span>Cross-validation against ChEMBL database</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold mb-3 text-green-700">Experimental Validation</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                  <span>Temperature scaling for probability calibration</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                  <span>Prospective validation on held-out test set</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                  <span>Domain applicability analysis</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2" />
                  <span>Uncertainty quantification methods</span>
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Data Sources and Citations */}
      <Card>
        <CardHeader>
          <CardTitle>Data Sources & Citations</CardTitle>
          <CardDescription>Original data sources and recommended citations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">Primary Source</h3>
              <p className="text-sm text-muted-foreground">
                BindingDB: A web-accessible database of measured binding affinities, focusing chiefly on the binding of small molecules to proteins.
              </p>
              <p className="text-xs mt-1 font-mono bg-gray-50 p-2 rounded">
                Gilson, M.K., Liu, T., Baitaluk, M., Nicola, G., Hwang, L., Chong, J. (2016). BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology. Nucleic Acids Research, 44(D1), D1045-D1053.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">Data Processing Pipeline</h3>
              <p className="text-sm text-muted-foreground">
                Custom preprocessing pipeline for kinase binding prediction including scaffold-based splitting and molecular standardization.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">Recommended Citation</h3>
              <p className="text-xs font-mono bg-gray-50 p-2 rounded">
                When using this model or dataset, please cite both the original BindingDB paper and acknowledge the preprocessing pipeline used for kinase binding prediction.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
