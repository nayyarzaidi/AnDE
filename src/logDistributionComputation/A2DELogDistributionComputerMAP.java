package logDistributionComputation;

import DataStructure.wdAnDEParameters;
import logDistributionComputation.LogDistributionComputerAnDE;
import Utils.SUtils;
import weka.core.Instance;

public class A2DELogDistributionComputerMAP extends LogDistributionComputerAnDE {

	public static LogDistributionComputerAnDE singleton = null;

	protected A2DELogDistributionComputerMAP(){}
	public static LogDistributionComputerAnDE getComputer() {
		if (singleton==null){
			singleton = new A2DELogDistributionComputerMAP();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnDEParameters params,Instance inst) {

		int N = params.getN();
		int n = params.getNAttributes();
		int nc = params.getNC();

		double probInitializerA2DE = Double.MAX_VALUE/((n+1)*(n+1));
		double[][][] spodeProbs = new double[n][][];
		int parentCount = 0;

		for (int up1 = 1; up1 < n; up1++) {
			int up2size = 0;
			for (int up2 = 0; up2 < up1; up2++) {
				up2size++;
			}
			spodeProbs[up1]  = new double[up2size][nc];
		}

		for (int up1 = 1; up1 < n; up1++) {
			int x_up1 = (int) inst.value(up1);

			for (int up2 = 0; up2 < up1; up2++) {
				int x_up2 = (int) inst.value(up2);

				long index = 0;
				int countOfX1AndX2AndY = 0;
				for (int c = 0; c < nc; c++) {
					index = params.getAttributeIndex(up1, x_up1, up2, x_up2, c);
					countOfX1AndX2AndY += params.getCountAtFullIndex(index);
				}

				// Check that attribute value has a frequency of m_Limit or greater
				if (countOfX1AndX2AndY >= SUtils.m_Limit) {
					parentCount++;

					for (int c = 0; c < nc; c++) {
						index = params.getAttributeIndex(up1, x_up1, up2, x_up2, c);
						spodeProbs[up1][up2][c] = probInitializerA2DE * SUtils.MEsti(params.getCountAtFullIndex(index), N, params.getParamsPetAtt(up1) * params.getParamsPetAtt(up2) * nc);						
					}
				}							
			}
		}

		// Check that atleast one parent is used, otherwise, do A1DE.
		if (parentCount < 1) {
			System.out.println("Resorting to A1DE");
			LogDistributionComputerAnDE A1DE = LogDistributionComputerAnDE.getDistributionComputer(1, 1);
			A1DE.compute(probs, params, inst);						
		} else {

			for (int up1 = 2; up1 < n; up1++) { // Parent1
				int x_up1 = (int) inst.value(up1);

				for (int up2 = 1; up2 < up1; up2++) { // Parent2
					int x_up2 = (int) inst.value(up2);

					for (int uc = 0; uc < up2; uc++) { // Child
						int x_uc = (int) inst.value(uc);

						for (int c = 0; c < nc; c++) {	// Class
							long index = params.getAttributeIndex(up1, x_up1, up2, x_up2, uc, x_uc, c);
							double parentFreq = params.getCountAtFullIndex(index);

							long index2 = params.getAttributeIndex(up1, x_up1, up2, x_up2, c);
							long index3 = params.getAttributeIndex(up2, x_up2, uc, x_uc, c);
							long index4 = params.getAttributeIndex(up1, x_up1, uc, x_uc, c);

							spodeProbs[up1][up2][c] *= SUtils.MEsti(parentFreq, params.getCountAtFullIndex(index2), params.getParamsPetAtt(uc));
							spodeProbs[up2][uc][c] *= SUtils.MEsti(parentFreq, params.getCountAtFullIndex(index3), params.getParamsPetAtt(up1));
							spodeProbs[up1][uc][c] *= SUtils.MEsti(parentFreq, params.getCountAtFullIndex(index4), params.getParamsPetAtt(up2));
						}					
					}
				}
			}

			/* add all the probabilities for each class */
			for (int c = 0; c < nc; c++) {
				for (int up1 = 1; up1 < n; up1++) {
					for (int up2 = 0; up2 < up1; up2++) {
						probs[c] += spodeProbs[up1][up2][c];
					}
				}			
			}

			SUtils.log(probs);
		}
	}

}
