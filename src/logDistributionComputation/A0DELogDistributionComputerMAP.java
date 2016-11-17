package logDistributionComputation;

import DataStructure.wdAnDEParameters;
import logDistributionComputation.LogDistributionComputerAnDE;
import Utils.SUtils;
import weka.core.Instance;

public class A0DELogDistributionComputerMAP extends LogDistributionComputerAnDE {

	public static LogDistributionComputerAnDE singleton = null;

	protected A0DELogDistributionComputerMAP(){}

	public static LogDistributionComputerAnDE getComputer() {
		if(singleton == null){
			singleton = new A0DELogDistributionComputerMAP();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnDEParameters params, Instance inst) {

		for (int c = 0; c < probs.length; c++) {
			probs[c] = Math.log(SUtils.MEsti(params.getCountAtFullIndex(c),params.getN(),params.getNC()));
		}

		for (int c = 0; c < probs.length; c++) {

			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				long index = params.getAttributeIndex(att1, att1val, c);				
				probs[c] += Math.log(SUtils.MEsti(params.getCountAtFullIndex(index), 
						params.getCountAtFullIndex(c), params.getParamsPetAtt(att1)));
			}
		}
	}

}
