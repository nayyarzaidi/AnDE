package DataStructure;

import java.util.concurrent.Callable;

import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class CallableWdAnDEParametersFlat implements Callable<Double> {

	private Instances data;
	private int start;
	private int stop;

	private double Error;
	private int[] XYCount;	

	private int numTuples;
	private wdAnDEParametersFlat dParameters;

	private int threadID;

	public CallableWdAnDEParametersFlat(int start, int stop, Instances data, int numTuples, int[] XYCount, wdAnDEParametersFlat dParameters, int th) {
		this.data = data;
		this.start = start;
		this.stop = stop;

		this.XYCount = XYCount;
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
			dParameters.incCountAtFullIndex(XYCount, x_C);

			if (numTuples == 0) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					int index = (int) dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.incCountAtFullIndex(XYCount, index);				
				}

			} else if (numTuples == 1) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					int index = (int) dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.incCountAtFullIndex(XYCount, index);	

					for (int u2 = 0; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						index = (int) dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
						dParameters.incCountAtFullIndex(XYCount, index);		
					}
				}

			} else if (numTuples == 2) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					int index = (int) dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.incCountAtFullIndex(index);		

					for (int u2 = 0; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						index = (int) dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
						dParameters.incCountAtFullIndex(index);	

						for (int u3 = 0; u3 < u2; u3++) {
							int x_u3 = (int) inst.value(u3);

							index = (int) dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
							dParameters.incCountAtFullIndex(XYCount, index);		
						}
					}
				}
			}

			numProcessed++;
			if ((numProcessed % SUtils.displayPerfAfterInstances) == 0) {
				//System.out.print(perfOutput.charAt(threadID));
				// Nayyar commenting following line for calling in streamAnDE
				//System.out.print(SUtils.perfOutput.charAt(threadID));
			}

		} // ends j

		return Error;
	}

}
