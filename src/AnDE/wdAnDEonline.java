/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi, Francois Petitjean and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * wdAnDE Classifier
 * 
 * wdAnDE.java     
 * Code written by: Nayyar Zaidi
 * 
 * Options:
 * -------
 * 
 * -D   Discretize numeric attributes
 * -V 	Verbosity
 * -M   Multi-threaded
 * 
 * -S	Structure learning (A1DE, A2DE)
 * -P	Parameter learning (MAP)
 * -I   Structure to use (Flat, Indexed, IndexedBig, BitMap) 
 * 
 */
package AnDE;

import DataStructure.wdAnDEParameters;
import DataStructure.wdAnDEParametersFlat;
import logDistributionComputation.LogDistributionComputerAnDE;

import Utils.SUtils;
import Utils.plTechniques;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

public class wdAnDEonline extends AbstractClassifier implements OptionHandler, UpdateableClassifier  {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	int[] paramsPerAtt;

	private String m_S = "A1DE"; 						// -S (A0DE, A1DE, A2DE)

	private boolean m_MVerb = false; 					// -V		

	private double[] probs;	

	private int numTuples;

	protected wdAnDEParameters dParameters_;
	private LogDistributionComputerAnDE logDComputer;


	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances.deleteWithMissingClass();
		nInstances = instances.numInstances();
		nAttributes = instances.numAttributes() - 1;		
		nc = instances.numClasses();

		probs = new double[nc];		

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = instances.attribute(u).numValues();
		}		

		/*
		 * Initialize structure array based on m_S
		 */
		if (m_S.equalsIgnoreCase("A0DE")) {
			// A0DE
			numTuples = 0;
		} else if (m_S.equalsIgnoreCase("A1DE")) {
			// A1DE			
			numTuples = 1;
		} else if (m_S.equalsIgnoreCase("A2DE")) {
			// A2DE			
			numTuples = 2;
		}

		/* 
		 * ----------------------------------------------------------------------------------------
		 * Start Parameter Learning Process
		 * ----------------------------------------------------------------------------------------
		 */

		int scheme = 1;

		/*
		 * ---------------------------------------------------------------------------------------------
		 * Intitialize data structure
		 * ---------------------------------------------------------------------------------------------
		 */

		scheme = plTechniques.MAP;			

		logDComputer = LogDistributionComputerAnDE.getDistributionComputer(numTuples, scheme);

		dParameters_ = new wdAnDEParametersFlat(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples,m_MVerb);				

		if (m_MVerb)
			System.out.println("All data structures are initialized. Starting to estimate parameters.");
		
		if (nInstances > 0) {
			for (int i = 0; i < nInstances; i++) {
				Instance instance = instances.instance(i);
				dParameters_.updateFirstPass(instance);				
			}
		}
	}

	public void updateClassifier(Instance instance) {
		dParameters_.updateFirstPass(instance);
		dParameters_.incrementN();
	}

	@Override
	public double[] distributionForInstance(Instance instance) {

		double[] probs = logDistributionForInstance(instance);
		SUtils.exp(probs);
		return probs;
	}	

	public double[] logDistributionForInstance(Instance inst) {
		double[] probs = new double[nc];
		logDistributionForInstance(probs,inst) ;
		return probs;
	}

	public void logDistributionForInstance(double [] probs,Instance inst) {
		logDComputer.compute(probs, dParameters_, inst);
		SUtils.normalizeInLogDomain(probs);
	}

	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {

		m_MVerb = Utils.getFlag('V', options);
		
		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_S = SK;
		}

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public static void main(String[] argv) {
		runClassifier(new wdAnDEonline(), argv);
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public int getnAttributes() {
		return nAttributes;
	}

	public wdAnDEParameters getdParameters_() {
		return dParameters_;
	}

	public Instances getM_Instances() {
		return m_Instances;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public String getMS() {
		return m_S;
	}

}
