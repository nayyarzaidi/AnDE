package Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class SUtils {

	public static int minNumThreads = 4000;
	public static int displayPerfAfterInstances = 1000;
	public static String perfOutput = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-+*`~!@#$%^&_|:;'?";
	public static int m_Limit = 1;


	public static int ind(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	public static double MEsti(double freq1, double freq2, double numValues) {
		double m_MEsti = 1.0;
		double mEsti = (freq1 + m_MEsti / numValues) / (freq2 + m_MEsti);
		return mEsti;
	}

	public static double MEsti(double freq1, double freq2) {
		double mEsti = freq1 / freq2;
		return mEsti;
	}

	public static void boundAndNormalizeInLogDomain(double[] logs,
			double maxDifference) {
		boundDifferences(logs, maxDifference);
		double logSum = sumInLogDomain(logs);
		for (int i = 0; i < logs.length; i++)
			logs[i] -= logSum;
	}

	public static void boundDifferences(double[] logs, double maxDifference) {
		double maxLog = logs[0];
		for (int i = 1; i < logs.length; i++) {
			if (maxLog < logs[i]) {
				maxLog = logs[i];
			}
		}
		for (int i = 0; i < logs.length; i++) {
			logs[i] = logs[i] - maxLog;
			if (logs[i] < -maxDifference) {
				logs[i] = -maxDifference;
			}
		}
	}

	public static void normalizeInLogDomain(double[] logs) {
		double logSum = sumInLogDomain(logs);
		for (int i = 0; i < logs.length; i++)
			logs[i] -= logSum;
	}

	public static double sumInLogDomain(double[] logs) {
		// first find max log value
		double maxLog = logs[0];
		int idxMax = 0;
		for (int i = 1; i < logs.length; i++) {
			if (maxLog < logs[i]) {
				maxLog = logs[i];
				idxMax = i;
			}
		}
		// now calculate sum of exponent of differences
		double sum = 0;
		for (int i = 0; i < logs.length; i++) {
			if (i == idxMax) {
				sum++;
			} else {
				sum += Math.exp(logs[i] - maxLog);
			}
		}
		// and return log of sum
		return maxLog + Math.log(sum);
	}

	public static void exp(double[] logs) {
		for (int c = 0; c < logs.length; c++) {
			logs[c] = Math.exp(logs[c]);
		}
	}

	public static void log(double[] logs) {
		for (int c = 0; c < logs.length; c++) {
			logs[c] = Math.log(logs[c]);
		}
	}	

	public static int[] sort(double[] mi) {		
		int[] sortedPositions = Utils.sort(mi);
		int n = mi.length;
		int[] order = new int[n];
		for (int i = 0; i < n; i++) {
			order[i] = sortedPositions[(n-1) - i];
		}
		return order;
	}

	public static int combination(int N, int k) {
		int n = 0;
		int num = factorial(N);
		int denum1 = factorial(N - k);
		int denum2 = factorial(k);
		n = (num) / (denum1 * denum2);
		return n;
	}

	public static int factorial(int a) {
		int facta = 1;		
		for (int i = a; i > 0; i--) {
			facta *= i;
		}		
		return facta;
	}

	public static int NC2(int a) {
		int count = 0;
		for (int att1 = 1; att1 < a; att1++) {			
			for (int att2 = 0; att2 < att1; att2++) {
				count++;	
			}
		}
		return count;		
	}

	public static int NC3(int a) {
		int count = 0;
		for (int att1 = 2; att1 < a; att1++) {			
			for (int att2 = 1; att2 < att1; att2++) {				
				for (int att3 = 0; att3 < att2; att3++) {
					count++;
				}
			}
		}
		return count;		
	}

	public static int NC4(int a) {
		int count = 0;

		for (int att1 = 3; att1 < a; att1++) {			
			for (int att2 = 2; att2 < att1; att2++) {				
				for (int att3 = 1; att3 < att2; att3++) {
					for (int att4 = 0; att4 < att3; att4++) {
						count++;
					}
				}
			}
		}
		return count;
	}

	public static int NC5(int a) {
		int count = 0;

		for (int att1 = 4; att1 < a; att1++) {			
			for (int att2 = 3; att2 < att1; att2++) {				
				for (int att3 = 2; att3 < att2; att3++) {
					for (int att4 = 1; att4 < att3; att4++) {
						for (int att5 = 0; att5 < att4; att5++) {
							count++;
						}
					}
				}
			}
		}
		return count;
	}

	public static void randomizeFast(int[] index, int nInstances) {
		for (int i = 0; i < nInstances; i++) {
			index[i] = i;
		}
		
		Random random = new Random(System.currentTimeMillis());
		
		for (int j = index.length - 1; j > 0; j-- ){
			int k = random.nextInt( j + 1 );
			int temp = index[j];
			index[j] = index[k];
			index[k] = temp;
		}
	}
	
	public final static void randomize(int[] att, int n) {

		Random random = new Random(System.currentTimeMillis());
		for (int i = 0; i < n; i++) 
			att[i] = -1;
		
		int j = 0;
		do {

			int k = random.nextInt(n);
			if (!inArray(k,att)) { 
				att[j] = k;
				j++;
			}

		} while (j < att.length);

		//		for (int i = 0; i < index.length; i++) {
		//			int k = random.nextInt(n);
		//			if (!inArray(k,index)) { 
		//				index[i] = k;
		//				j++;
		//			}
		//		}

	}

	public final static void randomize(int[] index) {
		Random random = new Random(System.currentTimeMillis());
		for (int j = index.length - 1; j > 0; j-- ){
			int k = random.nextInt( j + 1 );
			int temp = index[j];
			index[j] = index[k];
			index[k] = temp;
		}
	}

	public final void randomize(int[] index, Random random) {
		for (int j = index.length - 1; j > 0; j-- ){
			int k = random.nextInt( j + 1 );
			int temp = index[j];
			index[j] = index[k];
			index[k] = temp;
		}
	}

	public static void shuffleArray(int[] ar) {

		Random rnd = new Random();
		for (int i = ar.length - 1; i > 0; i--)
		{
			int index = rnd.nextInt(i + 1);
			// Simple swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
	}

	public static double maxAbsValueInAnArray(double[] array) {
		int index = 0;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (Math.abs(array[i]) > max) {
				max = Math.abs(array[i]);
				index = i;				
			}
		}		
		return Math.abs(array[index]);
	}

	public static int maxLocationInAnArray(double[] array) {
		int index = 0;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				index = i;				
			}
		}		
		return index;
	}

	public static int minLocationInAnArray(double[] array) {
		int index = 0;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] < min) {
				min = array[i];
				index = i;				
			}
		}		
		return index;
	}

	public static int findMaxValueLocationInNDMatrix(double[][] results, int dim) {
		double[] tempVector = new double[results.length];

		for (int i = 0; i < results.length; i++) {
			tempVector[i] = results[i][dim];
		}

		int index = minLocationInAnArray(tempVector);

		return index;
	}

	public static double MI(long[][] contingencyMatrix) {
		int n = 0;
		int nrows = contingencyMatrix.length;
		int ncols = contingencyMatrix[0].length;

		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += contingencyMatrix[r][c];
				colsSum[c] += contingencyMatrix[r][c];
				n += contingencyMatrix[r][c];
			}				
		}

		double MI = 0;

		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (contingencyMatrix[r][c] > 0) {
							double a = contingencyMatrix[r][c] / ( rowsSum[r]/(double)n * colsSum[c] ) ;
							MI += contingencyMatrix[r][c]/(double)n * Math.log(a / Math.log(2));
						}
					}
				}
			}
		}
		return MI;
	}

	public static double gsquare(long[][]observed){
		long n = 0;
		int nrows = observed.length;
		int ncols = observed[0].length;

		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += observed[r][c];
				colsSum[c] += observed[r][c];
				n += observed[r][c];
			}				
		}

		double gs = 0.0;
		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (observed[r][c] > 0) {
							double exp = (1.0*rowsSum[r]/n) * (1.0*colsSum[c]/n) ;
							gs+= 1.0*observed[r][c]*FastMath.log(observed[r][c]/exp);
						}
					}
				}
			}
		}
		gs*=2.0;
		return gs;
	}

	public static double chisquare(long[][]observed){
		long n = 0;
		int nrows = observed.length;
		int ncols = observed[0].length;

		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += observed[r][c];
				colsSum[c] += observed[r][c];
				n += observed[r][c];
			}				
		}

		double chi = 0.0;
		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (observed[r][c] > 0) {
							double exp = (1.0*rowsSum[r]/n) * (1.0*colsSum[c]/n) ;
							double diff = observed[r][c]-exp;
							chi+= diff*diff/exp;
						}
					}
				}
			}
		}
		return chi;
	}

	public static boolean inSet(int[] array, int element) {
		boolean present = false;

		if (array == null) {
			return false;
		}

		for (int i = 0; i < array.length; i++) {
			if (array[i] == element) {
				present = true;
				break;
			}
		}
		return present;
	}

	public static boolean linkExist(int[][] m_Parents, int[] m_Order, int i, int j) {
		boolean present = false;

		if (inSet(m_Parents[i], m_Order[j]))
			present = true;		

		if (inSet(m_Parents[j], m_Order[i]))
			present = true;

		return present;
	}

	public static int[] CheckForPerfectness(int[] m_TempParents, int[][] m_Parents, int[] m_Order) {
		int[] parents = null;		
		parents = new int[m_TempParents.length];

		int j = 0;
		for (int j1 = 0; j1 < m_TempParents.length; j1++) {
			for (int j2 = j1 + 1; j2 < m_TempParents.length; j2++) {

				if (SUtils.linkExist(m_Parents, m_Order, m_TempParents[j1], m_TempParents[j2])) {
					parents[j] = m_TempParents[j1];
					parents[j+1] = m_TempParents[j2];
					j+=2;
				}

			}
		}	

		return null;
	}

	public static void labelsToProbs(int[] labels, double[] probs) {
		int[] count = new int[probs.length];
		for (int i = 0; i < labels.length; i++) {
			count[labels[i]]++;
		}
		int maxCount = Integer.MIN_VALUE;
		int labelIndex = -1;
		for (int i = 0; i < count.length; i++) {
			if (count[i] > maxCount) {
				maxCount = count[i];
				labelIndex = i;
			}
		}

		probs[labelIndex] = 1.0;
	}

	public static boolean inArray(int val, int[] arr) {
		boolean flag = false;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == val) {
				flag = true;
			}
		}
		return flag;
	}

	public static int inArrayIndex(int val, int[] arr) {
		int index = 0;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == val) {
				index = i;
			}
		}
		return index;
	}

	public static double minAbsValueInAnArray(double[] array) {
		int index = 0;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (Math.abs(array[i]) < min) {
				min = Math.abs(array[i]);
				index = i;				
			}
		}		
		return Math.abs(array[index]);
	}

	public static double minNonZeroValueInAnArray(int[] array) {
		int index = 0;
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] != 0 && array[i] < min) {
				min = Math.abs(array[i]);
				index = i;				
			}
		}		
		return array[index];
	}

	public static Instances generateBaggedData(Instances instances) {
		int N = instances.numInstances();
		Random random = new Random(System.currentTimeMillis());

		Instances data = new Instances(instances, 0);

		for (int i = 0; i < N; i++) {
			int index = random.nextInt(N);
			data.add(instances.instance(index));
		}

		return data;
	}

	public static int noOfNonZeroElementsInAnArray(int[] array) {
		int counter = 0;
		for (int i = 0; i < array.length; i++) {
			if (array[i] == 0)
				counter++;
		}
		return  array.length - counter;
	}

	public static int[] getExistingClasses(int[] array) {
		int np = noOfNonZeroElementsInAnArray(array);
		int outArray[] = new int[np];
		int j = 0;
		for (int i = 0; i < array.length; i++) {
			if (array[i] != 0) {
				outArray[j] = array[i];
				j++;
			}
		}
		return outArray;
	}

	public static int[] getClasses(int[] array) {
		int np = noOfNonZeroElementsInAnArray(array);
		int outArray[] = new int[np];
		int j = 0;
		for (int i = 0; i < array.length; i++) {
			if (array[i] != 0) {
				outArray[j] = i;
				j++;
			}
		}
		return outArray;
	}

	public static int getLabelFromProbs(double[] probs) {
		int index = 0;
		double max_value = Double.MIN_VALUE;
		for (int i = 0; i < probs.length; i++) {
			if (probs[i] > max_value) {
				index = i;
				max_value = probs[i];
			}
		}
		return index;
	}

	public static Instances updateHeader(Instances instances, int[] classes) {
		Instances header = null;
		int nc = classes.length;
		int n = instances.numAttributes() - 1;
		ArrayList<Attribute> attlist = new ArrayList<Attribute>(n + 1);

		for (int i = 0; i < n; i++) {
			attlist.add(instances.attribute(i));
		}

		String className = instances.classAttribute().name();
		List<String> classNamesList = new ArrayList<String>(nc);
		for (int i = 0; i < nc; i++) {
			classNamesList.add(instances.attribute(instances.classIndex()).value(classes[i]));
		}
		Attribute classAtt = new Attribute(className, classNamesList);

		attlist.add(classAtt);

		header = new Instances(instances.relationName(), attlist, 0);
		header.setClassIndex(n);
		
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);
			inst.setClassValue(SUtils.inArrayIndex((int) inst.classValue(), classes));
			header.add(inst);
		}
		
		return header;
	}

	public static Instances getHeader(Instances instances, int[] classes) {
		Instances header = null;
		int nc = classes.length;
		int n = instances.numAttributes() - 1;
		ArrayList<Attribute> attlist = new ArrayList<Attribute>(n + 1);

		for (int i = 0; i < n; i++) {
			attlist.add(instances.attribute(i));
		}

		String className = instances.classAttribute().name();
		List<String> classNamesList = new ArrayList<String>(nc);
		for (int i = 0; i < nc; i++) {
			classNamesList.add(instances.attribute(instances.classIndex()).value(classes[i]));
		}
		Attribute classAtt = new Attribute(className, classNamesList);

		attlist.add(classAtt);

		header = new Instances(instances.relationName(), attlist, 0);
		header.setClassIndex(n);

		return header;
	}

	public static int findInClasses(String predstrLabel, String[] classes) {
		int index = 0;
		for (int c = 0; c < classes.length; c++) {
			if (predstrLabel.equalsIgnoreCase(classes[c])) 
				index = c;
		}
		return index;
	}

	public static double getGiniCoefficient(Instances instances) {
		double gc = 0.0;
		int nc = instances.numClasses();
		double localN = instances.numInstances();

		double[] classCount = new double[nc];
		for (int i = 0; i < localN; i++) {
			int x_C = (int) instances.instance(i).classValue();
			classCount[x_C]++;
		}

		for (int c = 0; c < nc; c++ ) {
			if (classCount[c] != 0) { 
				double pi = classCount[c]/localN;
				gc -= 1 * pi * (Math.log(pi) / Math.log(2));
			}
		}

		return gc;
	}
	

}


