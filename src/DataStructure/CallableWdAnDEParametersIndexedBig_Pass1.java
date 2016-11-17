package DataStructure;

import java.util.concurrent.Callable;

import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class CallableWdAnDEParametersIndexedBig_Pass1 implements Callable<Double> {

	private Instances data;
	private int start;
	private int stop;

	private double Error;
	//private BitSet[] combinationRequired;	

	private int numTuples;
	private wdAnDEParametersIndexedBig dParameters;
	
	private int threadID;

	public CallableWdAnDEParametersIndexedBig_Pass1(int start, int stop, Instances data, int numTuples, wdAnDEParametersIndexedBig dParameters, int th) {
		this.data = data;
		this.start = start;
		this.stop = stop;

		//this.combinationRequired = combinationRequired;
		this.numTuples = numTuples;
		this.dParameters = dParameters;
		
		this.threadID = th;
	}

	@Override
	public Double call() throws Exception {

		int n = dParameters.n;
		
		int numProcessed = 0;

		for (int j = start; j <= stop; j++) {
			Instance inst = data.instance(j);

			int x_C = (int) inst.classValue();
			dParameters.setCombinationRequired(x_C);

			if (numTuples == 0) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					long index = dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.setCombinationRequired(index);
				}

			} else if (numTuples == 1) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					long index = dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.setCombinationRequired(index);

					for (int u2 = 0; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
						dParameters.setCombinationRequired(index);
					}
				}

			} else if (numTuples == 2) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					long index = dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.setCombinationRequired(index);

					for (int u2 = 0; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
						dParameters.setCombinationRequired(index);

						for (int u3 = 0; u3 < u2; u3++) {
							int x_u3 = (int) inst.value(u3);

							index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
							dParameters.setCombinationRequired(index);
						}
					}
				}

			} else if (numTuples == 3) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					long index = dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.setCombinationRequired(index);

					for (int u2 = 0; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
						dParameters.setCombinationRequired(index);

						for (int u3 = 0; u3 < u2; u3++) {
							int x_u3 = (int) inst.value(u3);

							index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
							dParameters.setCombinationRequired(index);

							for (int u4 = 0; u4 < u3; u4++) {
								int x_u4 = (int) inst.value(u4);

								index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
								dParameters.setCombinationRequired(index);
							}
						}
					}
				}

			} else if (numTuples == 4) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					long index = dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.setCombinationRequired(index);

					for (int u2 = 0; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
						dParameters.setCombinationRequired(index);

						for (int u3 = 0; u3 < u2; u3++) {
							int x_u3 = (int) inst.value(u3);

							index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
							dParameters.setCombinationRequired(index);

							for (int u4 = 0; u4 < u3; u4++) {
								int x_u4 = (int) inst.value(u4);

								index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
								dParameters.setCombinationRequired(index);

								for (int u5 = 0; u5 < u4; u5++) {
									int x_u5 = (int) inst.value(u5);

									index = dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, x_C);
									dParameters.setCombinationRequired(index);									
								}
							}
						}
					}
				}
			}
			
			numProcessed++;
			if ((numProcessed % SUtils.displayPerfAfterInstances) == 0) {
				//System.out.print(perfOutput.charAt(threadID));
				System.out.print(SUtils.perfOutput.charAt(threadID));
			}

		} // ends j

		return Error;
	}

}
