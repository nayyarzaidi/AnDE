package DataStructure;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import Utils.plTechniques;

import weka.core.Instance;
import weka.core.Instances;

public class wdAnDEParametersFlat extends wdAnDEParameters {

	protected static int MAX_TAB_LENGTH = Integer.MAX_VALUE-8;
	private boolean m_MVerb;

	/**
	 * Constructor called by wdAnDE
	 */
	public wdAnDEParametersFlat(int n, int nc, int N, int[] in_ParamsPerAtt, int m_P, int numTuples, boolean m_MVerb) {

		super(n, nc, N, in_ParamsPerAtt, m_P, numTuples);
		
		this.m_MVerb = m_MVerb;
		
		if (m_MVerb) {
			System.out.print("In Constructor of wdAnDEParametersFlat(), ");
			System.out.print("Total number of parameters are: " + getTotalNumberParameters() + ", ");
			System.out.println("Maximum TAB length is: " + MAX_TAB_LENGTH + ".");
		}
		
		if (getTotalNumberParameters() > MAX_TAB_LENGTH) {
			System.err.println("CRITICAL ERROR: 'wdAnDEParametersFlat' not implemented for such dimensionalities. Use 'wdAnDEParametersIndexed'");
			System.exit(-1);
		}

		if (scheme == plTechniques.MAP) {			
			initCount(getTotalNumberParameters());			
		}
	}


	@Override
	public void updateFirstPass(Instance inst) {

		int x_C = (int) inst.classValue();
		xyCount[x_C]++;

		if (numTuples == 0) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);				
			}

		} else if (numTuples == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);	

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);		
				}
			}

		} else if (numTuples == 2) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);		

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);	

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						incCountAtFullIndex(index);		
					}
				}
			}
		}
	}

	@Override
	public void updateFirstPass_m(Instances m_Instances) {

		int nThreads;
		int minNPerThread = 4000;					
		int N = m_Instances.numInstances();

		int[][] threadXYCount;
		ExecutorService executor;

		if (N < minNPerThread) {
			nThreads = 1;
		} else {
			nThreads = Runtime.getRuntime().availableProcessors();
			if (N/nThreads < minNPerThread) {
				nThreads = N/minNPerThread + 1;
			}
		}
		
		if (m_MVerb) 
			System.out.println("In wdAnDEParametersFlat() - Pass 1: Launching " + nThreads + " threads");

		threadXYCount = new int[nThreads][(int)getTotalNumberParameters()];
		executor = Executors.newFixedThreadPool(nThreads);					

		Future<Double>[] futures = new Future[nThreads];

		int assigned = 0;
		int remaining = N;

		for (int th = 0; th < nThreads; th++) {
			/*
			 * Compute the start and stop indexes for thread th
			 */
			int start = assigned;
			int nInstances4Thread = remaining / (nThreads - th);
			assigned += nInstances4Thread;
			int stop = assigned - 1;
			remaining -= nInstances4Thread;

			/*
			 * Calling thread
			 */
			Callable<Double> thread = new CallableWdAnDEParametersFlat(start, stop, m_Instances, numTuples, threadXYCount[th], this, th);

			futures[th] = executor.submit(thread);
		}

		for (int th = 0; th < nThreads; th++) {

			try {
				double temp = futures[th].get();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}			

			for (int i = 0; i < xyCount.length; i++) {
				xyCount[i] += threadXYCount[th][i];
			}
		}

		executor.shutdown();
		
		if (m_MVerb) 
			System.out.println("In wdAnDEParametersFlat() - Pass 1: All threads finished.");		
	}

	@Override
	public void finishedFirstPass() {		
		// No need to convert counts into probabilities. This will be done at classification time.
	}

	@Override
	public boolean needSecondPass() {
		return false;
	}

	@Override
	public void update_MAP(Instance inst) {
		// Nothing to do, needSecondPass() is false.
	}

	@Override
	public void update_MAP_m(Instances m_Instances) {
		// Nothing to do, needSecondPass() is false.
	}	

	/* 
	 * --------------------------------------------------------------------------------
	 * Access Functions
	 * -------------------------------------------------------------------------------- 
	 */

	@Override
	public int getCountAtFullIndex(long index){
		return xyCount[(int)index];
	}	

	@Override
	protected void initCount(long size) {
		xyCount = new int[(int)size];
	}

	@Override
	public void setCountAtFullIndex(long index, int count) {
		xyCount[(int)index] = count;
	}

	@Override
	public void incCountAtFullIndex(long index, int value) {
		xyCount[(int)index] += value;
	}

	@Override
	public void incCountAtFullIndex(long index) {
		xyCount[(int)index]++;
	}

	public void setCountAtFullIndex(int[]tab, long index, int val) {
		tab[(int)index] = val;
	}

	public void incCountAtFullIndex(int[]tab, long index) {
		tab[(int)index]++;
	}

	@Override
	public long getTotalNumberParameters() {
		return np;
	}

} // ends class

