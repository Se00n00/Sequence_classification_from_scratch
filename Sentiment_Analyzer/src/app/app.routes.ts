import { Routes } from '@angular/router';
import { WildPageComponent } from './wild-page/wild-page.component';
import { HomeComponent } from './home/home.component'; // Import HomeComponent

export const routes: Routes = [
  { path: '', component: HomeComponent }, // Route for the home page
  { path: '**', component: WildPageComponent } // Wildcard route for 404
];
