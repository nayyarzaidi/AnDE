package DataStructure;

import DataStructure.indexTrie;
import weka.core.Instance;
import weka.core.Instances;

public abstract class wdAnDEParameters {

	protected long np;

	protected int n;
	protected int nc;
	protected int N;
	protected int scheme;

	protected int[] paramsPerAtt;	

	protected indexTrie[] indexTrie_;

	protected int [] xyCount;
	
	protected int numTuples;
	
	protected static int MAX_TAB_LENGTH = Integer.MAX_VALUE-8;	
	
	/**
	 * Constructor called by wdAnDE
	 */
	public wdAnDEParameters(int n, int nc, int N, int[] in_ParamsPerAtt, int m_P, int numTuples) {
		
		this.n = n;
		this.nc = nc;
		this.N = N;
		
		scheme = m_P;
		this.numTuples = numTuples;

		paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = in_ParamsPerAtt[u];
		}

		indexTrie_ = new indexTrie[n];				

		if (numTuples == 0) {
			np = nc;
			for (int u1 = 0; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();

				indexTrie_[u1].set(np);
				np += (paramsPerAtt[u1] * nc);
			}
		} else if (numTuples == 1) {
			np = nc;
			for (int u1 = 0; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].set(np);
				
				np += (paramsPerAtt[u1] * nc);
				
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 0; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].set(np);

					np += (paramsPerAtt[u1] * paramsPerAtt[u2] * nc);												
				}					
			}
		} else if (numTuples == 2) {
			np = nc;
			for (int u1 = 0; u1 < n; u1++) {
				
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].set(np);
				np += (paramsPerAtt[u1] * nc);
				
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 0; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].set(np);
					np += (paramsPerAtt[u1] * paramsPerAtt[u2] * nc);
					
					indexTrie_[u1].children[u2].children = new indexTrie[n];

					for (int u3 = 0; u3 < u2; u3++) {
						
						indexTrie_[u1].children[u2].children[u3] = new indexTrie();
						indexTrie_[u1].children[u2].children[u3].set(np);		

						np += (paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * nc);												
					}					
				}
			}
		} 
	}

	/**
	 * Function called in the first pass to look at the combinations that have been seen or not. 
	 * Then the function finishedUpdatingSeenObservations should be called, and then the update_MAP function. 
	 * @param inst
	 */
	public abstract void updateFirstPass(Instance inst);

	/**
	 * Multi-threaded version of updateFirstPass 
	 * @param inst
	 */
	public abstract void updateFirstPass_m(Instances m_Instances);

	/**
	 * Function called to initialize the counts, if needed, in the second pass after having called 'update_seen_observations' on every instance first.
	 * Needs to be overriden, or will just do nothing 
	 * @param inst
	 */
	public void update_MAP(Instance inst) {

	}

	/**
	 * Multi-threaded version of update_MAP 
	 * @param inst
	 */
	public void update_MAP_m(Instances m_Instances) {

	}

	/**
	 * Function called when the first pass is finished
	 */
	public abstract void finishedFirstPass();

	public abstract boolean needSecondPass();

	public abstract int getCountAtFullIndex(long index);
	
	public abstract void setCountAtFullIndex(long index,int count);
	
	public void incCountAtFullIndex(long index){
		incCountAtFullIndex(index,1);
	}
	
	public abstract void incCountAtFullIndex(long index,int value);
	
	protected abstract void initCount(long size);

	// ----------------------------------------------------------------------------------
	// Access Functions
	// ----------------------------------------------------------------------------------
	
	public int getClassIndex(int k) {
		return k;
	}

	public long getAttributeIndex(int att1, int att1val, int c) {
		long offset = indexTrie_[att1].offset;		
		return offset + c * (paramsPerAtt[att1]) + att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int c) {
		long offset = indexTrie_[att1].children[att2].offset;		
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].offset;
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) + 
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int att4, int att4val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].children[att4].offset;
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4]) +
				att4val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) +
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int att4, int att4val, int att5, int att5val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].children[att4].children[att5].offset;		
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4] * paramsPerAtt[att5]) +
				att5val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4]) +
				att4val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) +
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}
	
	public long getTotalNumberParameters() {
		return np;
	}
	
	public void printStatistics() {

		int[] countVector = new int[7];

		if (numTuples == 1) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 0; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						long index = getAttributeIndex(u1, u1val, c);
						int count = getCountAtFullIndex(index);
						if (count == 0) {
							countVector[0]++;
						} else if (count == 1) {
							countVector[1]++;
						} else if (count > 1 && count <= 5) {
							countVector[2]++;
						} else if (count > 5 && count <= 10) {
							countVector[3]++;
						} else if (count > 10 && count <= 15) {
							countVector[4]++;
						} else if (count > 15 && count <= 20) {
							countVector[5]++;
						} else if (count > 20) {
							countVector[6]++;
						}
					}
				}				
			}

		} else if (numTuples == 2) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 1; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 0; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

								long index = getAttributeIndex(u1, u1val, u2, u2val, c);
								int count = getCountAtFullIndex(index);
								if (count == 0) {
									countVector[0]++;
								} else if (count == 1) {
									countVector[1]++;
								} else if (count > 1 && count <= 5) {
									countVector[2]++;
								} else if (count > 5 && count <= 10) {
									countVector[3]++;
								} else if (count > 10 && count <= 15) {
									countVector[4]++;
								} else if (count > 15 && count <= 20) {
									countVector[5]++;
								} else if (count > 20) {
									countVector[6]++;
								}
							}
						}
					}
				}
			}	

		} else if (numTuples == 3) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 2; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 1; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 0; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {	

										long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
										int count = getCountAtFullIndex(index);
										if (count == 0) {
											countVector[0]++;
										} else if (count == 1) {
											countVector[1]++;
										} else if (count > 1 && count <= 5) {
											countVector[2]++;
										} else if (count > 5 && count <= 10) {
											countVector[3]++;
										} else if (count > 10 && count <= 15) {
											countVector[4]++;
										} else if (count > 15 && count <= 20) {
											countVector[5]++;
										} else if (count > 20) {
											countVector[6]++;
										}
									}
								}
							}
						}
					}
				}
			}

		} else if (numTuples == 4) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 3; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 2; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 1; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 0; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
												int count = getCountAtFullIndex(index);
												if (count == 0) {
													countVector[0]++;
												} else if (count == 1) {
													countVector[1]++;
												} else if (count > 1 && count <= 5) {
													countVector[2]++;
												} else if (count > 5 && count <= 10) {
													countVector[3]++;
												} else if (count > 10 && count <= 15) {
													countVector[4]++;
												} else if (count > 15 && count <= 20) {
													countVector[5]++;
												} else if (count > 20) {
													countVector[6]++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		} else if (numTuples == 5) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 4; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 3; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 2; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 1; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												for (int u5 = 0; u5 < u4; u5++) {
													for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {

														long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
														int count = getCountAtFullIndex(index);
														if (count == 0) {
															countVector[0]++;
														} else if (count == 1) {
															countVector[1]++;
														} else if (count > 1 && count <= 5) {
															countVector[2]++;
														} else if (count > 5 && count <= 10) {
															countVector[3]++;
														} else if (count > 10 && count <= 15) {
															countVector[4]++;
														} else if (count > 15 && count <= 20) {
															countVector[5]++;
														} else if (count > 20) {
															countVector[6]++;
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		double totalCount = 0;
		for (int i = 0; i < 7; i++) {
			totalCount += countVector[i];
		}

		System.out.println("Priniting Statistics");
		System.out.println(" = 0           :" + countVector[0] + " : " + countVector[0]/totalCount);
		System.out.println(" = 1           :" + countVector[1] + " : " + countVector[1]/totalCount);
		System.out.println(" > 1 && <= 5   :" + countVector[2] + " : " + countVector[2]/totalCount);
		System.out.println(" > 6 && <= 10  :" + countVector[3] + " : " + countVector[3]/totalCount);
		System.out.println(" > 11 && <= 15 :" + countVector[4] + " : " + countVector[4]/totalCount);
		System.out.println(" > 15 && <= 20 :" + countVector[5] + " : " + countVector[5]/totalCount);
		System.out.println(" > 20          :" + countVector[6] + " : " + countVector[6]/totalCount);		
	}

	public int getNAttributes(){
		return n;
	}
	
	public int getN(){
		return N;
	}
	
	public void incrementN(){
		 N++;
	}
	
	public int getNC(){
		return nc;
	}
	
	public int getParamsPetAtt(int att){
		return paramsPerAtt[att];
	}

} // ends class

