import java.util.*;

public class AnagramChecker {
    public static boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        char[] arr1 = s.toCharArray();
        char[] arr2 = t.toCharArray();
        Arrays.sort(arr1);
        Arrays.sort(arr2);
        return Arrays.equals(arr1, arr2);
    }

    public static void main(String[] args) {
        System.out.println(isAnagram("ready", "adyer")); // true
        System.out.println(isAnagram("bat", "bar"));     // false
    }
}
