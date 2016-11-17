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
import DataStructure.wdAnDEParametersIndexedBig;
import logDistributionComputation.LogDistributionComputerAnDE;

import Utils.SUtils;
import Utils.plTechniques;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.supervised.attribute.Discretize;

public class wdAnDE extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	int[] paramsPerAtt;

	private String m_S = "A1DE"; 						// -S (A0DE, A1DE, A2DE)
	private String m_P = "MAP";  						// -P (MAP)
	private String m_I = "Flat"; 						// -I (Flat, Indexed, IndexedBig, BitMap)

	private boolean m_Discretization = false; 			// -D
	private boolean m_MVerb = false; 					// -V		
	private boolean m_MultiThreaded = false; 			// -M

	private double[] probs;	
	private int numTuples;

	private Discretize m_Disc = null;
	protected wdAnDEParameters dParameters_;
	private LogDistributionComputerAnDE logDComputer;

	private boolean m_MThreadVerb = false;	
	
	@Override
	public void buildClassifier(Instances instances) throws Exception {

		Instances  m_DiscreteInstances = null;
		
		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new weka.filters.supervised.attribute.Discretize();
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(instances);
			System.out.println("Applying Discretization Filter");
			m_DiscreteInstances = weka.filters.Filter.useFilter(instances, m_Disc);
			System.out.println("Done");

			m_Instances = new Instances(m_DiscreteInstances);
			m_DiscreteInstances = new Instances(m_DiscreteInstances, 0);
		} else {
			m_Instances = new Instances(instances);
			instances = new Instances(instances, 0);
		}

		// remove instances with missing class
		m_Instances.deleteWithMissingClass();
		nInstances = m_Instances.numInstances();
		nAttributes = m_Instances.numAttributes() - 1;		
		nc = m_Instances.numClasses();

		probs = new double[nc];		

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
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

		if (m_P.equalsIgnoreCase("MAP")) {	
			/*
			 * MAP - Maximum Likelihood Estimates of the Parameters characterzing P(x_i|y)
			 */
			scheme = plTechniques.MAP;			

		} else {
			System.out.println("m_P value should be from set {MAP}");
		}

		logDComputer = LogDistributionComputerAnDE.getDistributionComputer(numTuples, scheme);

		if (m_I.equalsIgnoreCase("Flat")) {
			dParameters_ = new wdAnDEParametersFlat(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples,m_MVerb);				
		} else if (m_I.equalsIgnoreCase("Indexed")) {
		} else if (m_I.equalsIgnoreCase("IndexedBig")) {
			dParameters_ = new wdAnDEParametersIndexedBig(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples);	
		} else if (m_I.equalsIgnoreCase("BitMap")) {
		} else {
			System.out.println("m_I value should be from set {Flat, Indexed, IndexedBig, BitMap}");
		}		

		/*
		 * ---------------------------------------------------------------------------------------------
		 * Create Data Structure by leveraging ONE or TWO pass through the data
		 * (These routines are common to all parameter estimation methods)
		 * ---------------------------------------------------------------------------------------------
		 */		
		if (m_MultiThreaded) {

			dParameters_.updateFirstPass_m(m_Instances);
			
			if (m_MVerb)
				System.out.println("Finished first pass.");

			dParameters_.finishedFirstPass();

			if (dParameters_.needSecondPass() ){
				dParameters_.update_MAP_m(m_Instances);	
				
				if (m_MVerb)
					System.out.println("Finished second pass.");
			}

		} else {

			for (int i = 0; i < nInstances; i++) {
				Instance instance = m_Instances.instance(i);
				dParameters_.updateFirstPass(instance);				
			}
			if (m_MVerb)
				System.out.println("Finished first pass.");

			dParameters_.finishedFirstPass();

			if (dParameters_.needSecondPass() ){
				for (int i = 0; i < nInstances; i++) {
					Instance instance = m_Instances.instance(i);
					dParameters_.update_MAP(instance);				
				}
				if (m_MVerb)
					System.out.println("Finished second pass.");
			}
		}

		/*
		 * Routine specific operations.
		 */

		if (m_MVerb)
			System.out.println("All data structures are initialized. Starting to estimate parameters.");

		// free up some space
		m_Instances = new Instances(m_Instances, 0);
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		
		if (m_Discretization) {
			synchronized(m_Disc) {	
				m_Disc.input(instance);
				instance = m_Disc.output();
			}
		}
		
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

		m_Discretization = Utils.getFlag('D', options);
		m_MVerb = Utils.getFlag('V', options);

		m_MultiThreaded = Utils.getFlag('M', options);
		m_MThreadVerb = Utils.getFlag('T', options);

		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_S = SK;
		}

		String MP = Utils.getOption('P', options);
		if (MP.length() != 0) {
			// m_P = Integer.parseInt(MP);
			m_P = MP;
		}

		String MI = Utils.getOption('I', options);
		if (MI.length() != 0) {
			m_I = MI;
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
		runClassifier(new wdAnDE(), argv);
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
	
	public boolean isM_MThreadVerb() {
		return m_MThreadVerb;
	}

}
