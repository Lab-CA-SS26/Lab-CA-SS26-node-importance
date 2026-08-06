#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

struct TauPair {
    double x, y;
};

inline long long merge(std::vector<double>& arr, std::vector<double>& temp, int left, int mid, int right) {
    int i = left, j = mid, k = left;
    long long inv_count = 0;
    while ((i <= mid - 1) && (j <= right)) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            inv_count = inv_count + (mid - i);
        }
    }
    while (i <= mid - 1) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (i = left; i <= right; i++) arr[i] = temp[i];
    return inv_count;
}

inline long long mergeSort(std::vector<double>& arr, std::vector<double>& temp, int left, int right) {
    long long inv_count = 0;
    if (right > left) {
        int mid = (right + left) / 2;
        inv_count += mergeSort(arr, temp, left, mid);
        inv_count += mergeSort(arr, temp, mid + 1, right);
        inv_count += merge(arr, temp, left, mid + 1, right);
    }
    return inv_count;
}

inline double kendall_tau(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    if (n <= 1) return 1.0;
    
    std::vector<TauPair> pairs(n);
    for (int i = 0; i < n; i++) {
        pairs[i] = {x[i], y[i]};
    }
    
    // Sort by x, then by y
    std::sort(pairs.begin(), pairs.end(), [](const TauPair& a, const TauPair& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });
    
    std::vector<double> y_sorted(n);
    for (int i = 0; i < n; i++) {
        y_sorted[i] = pairs[i].y;
    }
    
    std::vector<double> temp(n);
    long long inversions = mergeSort(y_sorted, temp, 0, n - 1);
    
    // Calculate ties
    long long ties_x = 0;
    long long ties_y = 0;
    long long ties_xy = 0;
    
    long long current_tie_x = 1;
    for (int i = 1; i < n; i++) {
        if (pairs[i].x == pairs[i-1].x) {
            current_tie_x++;
        } else {
            ties_x += current_tie_x * (current_tie_x - 1) / 2;
            current_tie_x = 1;
        }
    }
    ties_x += current_tie_x * (current_tie_x - 1) / 2;
    
    // sort by y to count ties in y
    std::sort(pairs.begin(), pairs.end(), [](const TauPair& a, const TauPair& b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });
    
    long long current_tie_y = 1;
    long long current_tie_xy = 1;
    for (int i = 1; i < n; i++) {
        if (pairs[i].y == pairs[i-1].y) {
            current_tie_y++;
            if (pairs[i].x == pairs[i-1].x) {
                current_tie_xy++;
            } else {
                ties_xy += current_tie_xy * (current_tie_xy - 1) / 2;
                current_tie_xy = 1;
            }
        } else {
            ties_y += current_tie_y * (current_tie_y - 1) / 2;
            current_tie_y = 1;
            ties_xy += current_tie_xy * (current_tie_xy - 1) / 2;
            current_tie_xy = 1;
        }
    }
    ties_y += current_tie_y * (current_tie_y - 1) / 2;
    ties_xy += current_tie_xy * (current_tie_xy - 1) / 2;
    
    long long n0 = (long long)n * (n - 1) / 2;
    
    // tau-b formula
    double num = n0 - ties_x - ties_y + ties_xy - 2 * inversions;
    double den = std::sqrt((double)(n0 - ties_x) * (n0 - ties_y));
    
    if (den == 0.0) return NAN;
    return num / den;
}
