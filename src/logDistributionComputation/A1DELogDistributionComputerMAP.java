package logDistributionComputation;

import DataStructure.wdAnDEParameters;
import logDistributionComputation.LogDistributionComputerAnDE;
import Utils.SUtils;
import weka.core.Instance;

public class A1DELogDistributionComputerMAP extends LogDistributionComputerAnDE {

	public static LogDistributionComputerAnDE singleton = null;

	protected A1DELogDistributionComputerMAP(){}

	public static LogDistributionComputerAnDE getComputer() {
		if(singleton == null){
			singleton = new A1DELogDistributionComputerMAP();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnDEParameters params, Instance inst) {

		int N = params.getN();
		int n = params.getNAttributes();
		int nc = params.getNC();

		double probInitializerA1DE = Double.MAX_VALUE/(n+1);
		double[][] spodeProbs = new double[n][nc];
		int parentCount = 0;

		for (int up = 0; up < n; up++) {
			int x_up = (int) inst.value(up);

			long index = 0;
			int countOfX1AndY = 0;
			for (int c = 0; c < nc; c++) {
				index = params.getAttributeIndex(up, x_up, c);
				countOfX1AndY += params.getCountAtFullIndex(index);
			}

			// Check that attribute value has a frequency of m_Limit or greater
			if (countOfX1AndY > SUtils.m_Limit) {
				parentCount++;

				for (int c = 0; c < nc; c++) {
					index = params.getAttributeIndex(up, x_up, c);
					spodeProbs[up][c] = probInitializerA1DE * SUtils.MEsti(params.getCountAtFullIndex(index), N, params.getParamsPetAtt(up) * nc);
				}
			}							
		}

		// Check that atleast one parent is used, otherwise, do naive Bayes.
		if (parentCount < 1) {
			System.out.println("Resorting to NB");			
			LogDistributionComputerAnDE A0DE = LogDistributionComputerAnDE.getDistributionComputer(0, 1);
			A0DE.compute(probs, params, inst);
		} else {

			for (int up = 1; up < n; up++) { // Parent
				int x_up = (int) inst.value(up);

				for (int uc = 0; uc < up; uc++) { // Child				
					int x_uc = (int) inst.value(uc);

					for (int c = 0; c < nc; c++) {
						long index1 = params.getAttributeIndex(up, x_up, uc, x_uc, c);
						long index2 = params.getAttributeIndex(uc, x_uc, c);
						long index3 = params.getAttributeIndex(up, x_up, c);

						spodeProbs[uc][c] *= SUtils.MEsti(params.getCountAtFullIndex(index1), params.getCountAtFullIndex(index2), params.getParamsPetAtt(up));
						spodeProbs[up][c] *= SUtils.MEsti(params.getCountAtFullIndex(index1), params.getCountAtFullIndex(index3), params.getParamsPetAtt(uc));
					}					
				}
			}

			/* add all the probabilities for each class */
			for (int c = 0; c < nc; c++) {
				for (int u = 0; u < n; u++) {
					probs[c] += spodeProbs[u][c];
				}			
			}

			SUtils.log(probs);
		}
	}
}
