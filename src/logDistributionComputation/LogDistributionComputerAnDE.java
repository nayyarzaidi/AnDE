package logDistributionComputation;

import DataStructure.wdAnDEParameters;
import Utils.plTechniques;
import logDistributionComputation.A0DELogDistributionComputerMAP;
import logDistributionComputation.A1DELogDistributionComputerMAP;
import logDistributionComputation.A2DELogDistributionComputerMAP;
import weka.core.Instance;

public abstract class LogDistributionComputerAnDE {
	
	public abstract void compute(double[] probs, wdAnDEParameters params, Instance inst);
	
	public static LogDistributionComputerAnDE getDistributionComputer(int n, int scheme) {
		switch (n) {
		case 0: return getComputerA0DE(scheme);
		case 1: return getComputerA1DE(scheme);
		case 2: return getComputerA2DE(scheme);
		default:
			System.err.println("A"+n+"DE not implemented, choosing A0DE");
			return getComputerA1DE(scheme); 
		}
	}

	public static LogDistributionComputerAnDE getComputerA0DE(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return A0DELogDistributionComputerMAP.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return A0DELogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerAnDE getComputerA1DE(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return A1DELogDistributionComputerMAP.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return A1DELogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerAnDE getComputerA2DE(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return A2DELogDistributionComputerMAP.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return A2DELogDistributionComputerMAP.getComputer();
		}
	}
	
}
