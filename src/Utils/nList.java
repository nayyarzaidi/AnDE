package Utils;

import java.util.Random;

public class nList {
	
	int[] array;
	int index;
	int label;
	
	public nList(int length, int label) {
		index = 0;
		array = new int[length];
		this.label = label;
	}
	
	public void add(int val) {
		array[index] = val;
		index++;
	}
	
	public int get(int pos) {
		return array[pos];
	}
	
	public int size() {
		return array.length;
	}
	
	public void randomize() {
		Random random = new Random(System.currentTimeMillis());
		
		for (int i = array.length - 1; i > 0; i--) {
			int k = random.nextInt(i+1);
			int temp = array[k];
			array[k] = array[i];
			array[i] = temp;
		}
	}
	
	public boolean isEmpty() {
		if (array.length == 0)
			return false;
		else 
			return true;
	}

}
