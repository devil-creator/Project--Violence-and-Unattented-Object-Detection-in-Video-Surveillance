from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from kivy.clock import Clock
from datetime import datetime

class HomeScreen(Screen):
    pass

class MainApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.home_screen = HomeScreen(name="home")
        self.sm.add_widget(self.home_screen)

        # Load the KV file
        Builder.load_file("home_screen_1.kv")  # Ensure filename matches

        # Start updating time
        Clock.schedule_interval(self.update_time, 1)
        return self.sm

    def get_current_time(self):
        return datetime.now().strftime("%H:%M:%S")

    def update_time(self, dt):
        if self.sm.current_screen:
            time_label = self.sm.current_screen.ids.get("time_label")
            if time_label:
                time_label.text = self.get_current_time()

    def search_function(self):
        print("Search button clicked!")

    def view_live_detections(self):
        print("Viewing live detections!")

    def show_threat_details(self, threat_name):
        print(f"Showing details for: {threat_name}")


if __name__ == "__main__":
    MainApp().run()
