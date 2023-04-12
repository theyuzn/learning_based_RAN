#include <iostream>
#include <vector>

using namespace std;

struct UE {
    int id; 
    float data_to_transmit; 
    float channel_quality; 
};

// Define a class for scheduling radio resources to users in the system
class Scheduler {
public:
    Scheduler(int time_slot) {
        this->time_slot = time_slot;
        this->current_user = 0;
    }
    
    // Add a user to the system
    void add_user(UE user) {
        users.push_back(user);
    }
    
    // Schedule radio resources to the next user in the predefined order
    void schedule_next_user() {
        UE user = users[current_user];
        allocate_radio_resources(user.user_id, user.data_to_transmit, time_slot);
        current_user = (current_user + 1) % users.size();
    }
    
    // Allocate radio resources to a user
    void allocate_radio_resources(int user_id, float data_to_transmit, int time_slot) {
        // Implementation of radio resource allocation goes here
        cout << "Allocating radio resources to user " << user_id << " for " << time_slot << " ms" << endl;
    }
    
    // Wait for the duration of a time slot
    void wait_for_time_slot_duration() {
        // Implementation of time slot duration waiting goes here
    }
    
private:
    int time_slot; // Time slot allocated to each user (in milliseconds)
    int current_user; // Index of the current user being scheduled
    vector<UE> users; // List of users in the system
};

int main() {
    // Initialize the scheduler with a time slot of 1 millisecond
    Scheduler scheduler(1);
    
    // Add some users to the system
    UE user1 = {1, 10.0, 0.8};
    UE user2 = {2, 20.0, 0.6};
    UE user3 = {3, 15.0, 0.7};
    scheduler.add_user(user1);
    scheduler.add_user(user2);
    scheduler.add_user(user3);
    
    // Schedule radio resources to users in a round-robin manner
    while(true) {
        scheduler.schedule_next_user();
        if(scheduler.current_user == 0) {
            scheduler.wait_for_time_slot_duration();
        }
    }
    
    return 0;
}

